# api/chat.py
import os, json, base64
from flask import Flask, request, jsonify
from openai import OpenAI
from cryptography.fernet import Fernet

import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from guardrails import system_rules_text, build_profile_terms, judge_response
from context_selector import select_context

app = Flask(__name__)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

ALLOWED = {o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}

def with_cors(resp, origin: str):
    # If ALLOWED_ORIGINS is empty, DO NOT allow (safer)
    if origin and origin in ALLOWED:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def load_about_me():
    # Prefer encrypted if present
    enc = os.getenv("ABOUT_ME_JSON_ENC")
    if enc:
        key = os.environ["PROFILE_FERNET_KEY"].encode()
        f = Fernet(key)
        ciphertext = base64.b64decode(enc.encode("utf-8"))
        return json.loads(f.decrypt(ciphertext).decode("utf-8"))

    raw = os.environ.get("ABOUT_ME_JSON_BASE64", "")
    decoded = base64.b64decode(raw.encode("utf-8")).decode("utf-8")
    return json.loads(decoded)

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    origin = request.headers.get("Origin", "")

    # Handle CORS preflight
    if request.method == "OPTIONS":
        resp = jsonify({})
        resp = with_cors(resp, origin)
        return resp, 204

    # Load profile per request so env changes are reflected safely
    try:
        ABOUT_ME = load_about_me()
    except Exception as e:
        resp = jsonify({"error": f"ABOUT_ME load failed: {type(e).__name__}: {str(e)[:120]}"})
        resp = with_cors(resp, origin)
        return resp, 500

    CAPABILITIES = set(ABOUT_ME.get("capabilities", []))
    POLICY = ABOUT_ME.get("policy", {})
    PROFILE_TERMS = build_profile_terms(ABOUT_ME)

    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        resp = jsonify({"error": "message required"})
        resp = with_cors(resp, origin)
        return resp, 400

    history = data.get("history") or []
    validated_history = []
    if isinstance(history, list):
        for item in history[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                validated_history.append({"role": role, "content": content.strip()})

    system_text = system_rules_text(CAPABILITIES)
    context = select_context(msg, ABOUT_ME)

    messages = [{"role": "system", "content": system_text}]
    if validated_history:
        messages.extend(validated_history)
    messages.append({
        "role": "user",
        "content": (
            "BACKGROUND (selected sections):\n" + context + "\n\n" +
            "USER MESSAGE:\n" + msg
        )
    })

    try:
        r = client.chat.completions.create(
            model="gpt-5-nano",
            reasoning_effort="low",
            messages=messages,
        )
        draft = r.choices[0].message.content
    except Exception as e:
        resp = jsonify({"error": f"OpenAI error: {type(e).__name__}: {str(e)[:160]}"})
        resp = with_cors(resp, origin)
        return resp, 500

    decision, reason, suggest = judge_response(client, msg, draft, POLICY, CAPABILITIES, PROFILE_TERMS)
    final = draft if decision in {"ALLOW", "ERROR"} else (suggest or "I can’t answer that based on my profile.")

    resp = jsonify({"reply": final, "decision": decision})
    resp = with_cors(resp, origin)
    return resp
