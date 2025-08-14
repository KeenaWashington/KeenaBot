# api/chat.py
import os, json, base64
from flask import Flask, request, jsonify
from openai import OpenAI

from guardrails import build_guardrails, system_rules_text
from context_selector import select_context

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ABOUT_ME = json.loads(base64.b64decode(os.environ["ABOUT_ME_JSON_BASE64"]).decode("utf-8"))

CAPABILITIES = set(ABOUT_ME.get("capabilities", []))
POLICY = ABOUT_ME.get("policy", {})
guard = build_guardrails(CAPABILITIES, POLICY)

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400

    blocked, reason = guard(msg)
    if blocked:
        return jsonify({"reply": reason})

    system_text = system_rules_text(CAPABILITIES)
    context = select_context(msg, ABOUT_ME)

    r = client.chat.completions.create(
        model="gpt-5-mini", reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": f"BACKGROUND:\n{context}\n\nUSER:\n{msg}"}
        ],
    )
    return jsonify({"reply": r.choices[0].message.content})