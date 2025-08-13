import re

def build_guardrails(capabilities: set, policy: dict):
    refusals = policy.get("refusal_messages", {}) if policy else {}
    refuse_code = refusals.get(
        "code_help",
        "I don’t provide programming help or code fixes. I’m happy to talk about my background, projects, personal life, or preferences instead."
    )
    refuse_outside = refusals.get(
        "capability_outside",
        "I can’t do that. I only answer based on the info in my profile."
    )

    code_keywords = [
        r"\b(debug|fix|optimi[sz]e|refactor|rewrite|patch|trace|stack trace|error|exception)\b",
        r"\b(npm install|pip install|docker build|docker run|git clone|gradle|maven|dotnet build)\b",
        r"```",  # code fences
        r"\.(py|js|ts|jsx|tsx|cs|java|go|rs|php|rb|sql|json|yml|yaml|xml|html|css)\b",
    ]
    code_pattern = re.compile("|".join(code_keywords), re.I)
    request_verbs = re.compile(r"\b(build|fix|debug|implement|deploy|connect|integrate|code|program|script|refactor)\b", re.I)

    def is_code_request(text: str) -> bool:
        return bool(code_pattern.search(text or ""))

    def is_outside_capabilities(text: str) -> bool:
        return bool(request_verbs.search(text or ""))

    def guard(user_text: str):
        if is_code_request(user_text):
            return True, refuse_code
        if is_outside_capabilities(user_text):
            return True, refuse_outside
        return False, ""

    return guard

def system_rules_text(capabilities: set) -> str:
    caps = ", ".join(sorted(capabilities)) or "(none)"
    return (
        "You are KeenaBot, an AI persona inspired by Keena Washington.\n"
        "RULES:\n"
        "• Do NOT provide programming help, code, debugging, or technical troubleshooting.\n"
        "• Only discuss topics present in about_me.json.\n"
        f"• Only claim capabilities explicitly listed in about_me.json.capabilities: {caps}\n"
        "• If a request is outside those capabilities or not represented in the JSON, politely refuse and offer allowed topics.\n"
        "• Be concise, warm, and practical. No impersonation disclaimers; write as KeenaBot.\n"
    )