# api/chat.py
import os, json, base64
from flask import Flask, request, jsonify
from openai import OpenAI
from cryptography.fernet import Fernet
import re

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

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Support encrypted or plaintext profile JSON from env
enc = os.getenv("ABOUT_ME_JSON_ENC")
if enc:
    # Encrypted path: ABOUT_ME_JSON_ENC is base64 of Fernet-encrypted bytes
    key = os.environ["PROFILE_FERNET_KEY"].encode()
    f = Fernet(key)
    ciphertext = base64.b64decode(enc.encode())
    ABOUT_ME = json.loads(f.decrypt(ciphertext).decode("utf-8"))
else:
    # Plaintext path: ABOUT_ME_JSON_BASE64 is base64 of the raw JSON
    ABOUT_ME = json.loads(base64.b64decode(os.environ["ABOUT_ME_JSON_BASE64"]).decode("utf-8"))

CAPABILITIES = set(ABOUT_ME.get("capabilities", []))
POLICY = ABOUT_ME.get("policy", {})

WELCOME_MESSAGE = os.getenv(
    "WELCOME_MESSAGE",
    "Hello! I am an AI chatbot designed to respond as Keena would with a good amount of information about anything you could want to know. Feel free to ask anything about me, my life, or my work experience."
)

MODEL = os.getenv("MODEL", "gpt-5-mini")
REASONING = os.getenv("REASONING_EFFORT", "low")

PROFILE_TERMS = build_profile_terms(ABOUT_ME)

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

ALLOWED = {o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}

CRISIS_RE = re.compile(
    r"(?i)\b("
    r"kill myself|suicide|self[-\s]?harm|end my life|want to die|"
    r"going to kill myself|hurt myself|i want to hurt myself|unalive|take my life"
    r")\b"
)

def with_cors(resp, origin: str):
    # Only set CORS headers if the request Origin is in the allowed list
    if origin in ALLOWED:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    origin = request.headers.get("Origin", "")
    if request.method == "OPTIONS":  # CORS preflight
        return with_cors(jsonify({}), origin), 204
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    first = bool(data.get("first"))
    if not msg and not first:
        resp = jsonify({"error": "message required"})
        return with_cors(resp, origin), 400
    if first:
        resp = jsonify({"reply": WELCOME_MESSAGE, "decision": "ALLOW"})
        return with_cors(resp, origin)

    # Crisis short-circuit (no LLM call)
    if CRISIS_RE.search(msg):
        s = ABOUT_ME.get("suicide")
        if isinstance(s, dict):
            text = s.get("message") or s.get("text") or ""
        else:
            text = str(s or "")
        if not text:
            text = (
                "I'm sorry you must have mistaken me for someone who cares... womp womp ;( \n Now ask questions about me or go somewhere else you tricky wolf."
            )
        resp = jsonify({"reply": text, "decision": "CRISIS"})
        return with_cors(resp, origin)

    history = data.get("history") or []
    validated_history = []
    if isinstance(history, list):
        for item in history[-12:]:  # cap to last 12 items
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
        ),
    })

    try:
        r = client.chat.completions.create(
            model="gpt-5-mini",
            reasoning_effort="low",
            messages=messages,
        )
        draft = r.choices[0].message.content
    except Exception as e:
        resp = jsonify({"error": str(e)})
        return with_cors(resp, origin), 500

    decision, reason, suggest = judge_response(client, msg, draft, POLICY, CAPABILITIES, PROFILE_TERMS)
    if decision in {"ALLOW", "ERROR"}:
        final = draft
    else:
        final = suggest or "I can’t answer that based on my profile."

    resp = jsonify({"reply": final, "decision": decision})
    return with_cors(resp, origin)