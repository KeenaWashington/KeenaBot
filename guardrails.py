from __future__ import annotations
import json
import re
from typing import Dict, Set, Tuple

# -------------------- Persona/system rules --------------------

def system_rules_text(capabilities: Set[str]) -> str:
    caps = ", ".join(sorted(capabilities)) or "(none)"
    return (
        "You are KeenaBot, an AI persona inspired by Keena Washington.\n"
        "RULES:\n"
        "• Stay within the information provided in about_me.json.\n"
        f"• Only claim capabilities explicitly listed: {caps}\n"
        "• If a user asks for actions you can't perform, say you can’t do that.\n"
        "• If asked about a skill/preference not in your profile, say you don’t have that info.\n"
        "• Be concise, warm, and practical. No impersonation disclaimers; write as KeenaBot.\n"
        "• If asked to create, code, help create or code. Then refuse\n"
    )

# -------------------- Profile term builder --------------------

def _to_terms(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_to_terms(v))
        return out
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            out.extend(_to_terms(v))
        return out
    return []


def build_profile_terms(about_me: Dict) -> Dict[str, Set[str]]:
    """Flatten skills/preferences into two sets for quick checking and to
    condition the judge with a compact lexicon.
    Returns {"skills": set(str), "preferences": set(str)}
    """
    skills = set()
    prefs = set()

    skills_obj = (about_me or {}).get("skills", {})
    for key in [
        "languages",
        "frameworks_platforms",
        "databases",
        "cloud",
        "data",
        "api_formats",
        "ui_ux",
        "tools",
    ]:
        skills.update(s.strip() for s in _to_terms(skills_obj.get(key)) if isinstance(s, str))

    personal = (about_me or {}).get("personal", {})
    prefs.update(s.strip() for s in _to_terms(personal.get("hobbies")) if isinstance(s, str))
    favs = (personal or {}).get("favorites", {})
    prefs.update(s.strip() for s in _to_terms(favs.get("color")) if isinstance(s, str))
    prefs.update(s.strip() for s in _to_terms(favs.get("foods")) if isinstance(s, str))

    # Also collect websites, certifications, education keywords as auxiliary skills
    certs = (about_me or {}).get("certifications", [])
    skills.update(s.strip() for s in _to_terms(certs) if isinstance(s, str))
    edu = (about_me or {}).get("education", [])
    skills.update(s.strip() for s in _to_terms(edu) if isinstance(s, str))

    # Normalize to lower-case comparison set (but keep originals for display)
    def norm_set(xs):
        return {x.lower() for x in xs if isinstance(x, str) and x.strip()}

    return {
        "skills": norm_set(skills),
        "preferences": norm_set(prefs),
    }

# -------------------- Judge (LLM) --------------------

def judge_response(
    client,
    user_text: str,
    draft_answer: str,
    policy: Dict,
    capabilities: Set[str],
    profile_terms: Dict[str, Set[str]] | None,
    model: str = "gpt-5-nano",
) -> Tuple[str, str, str]:
    """Use a small model to judge if the draft should be sent as-is.

    Returns (decision, reason, suggest_reply), where decision is one of:
      - ALLOW — send draft as-is
      - OUT_OF_SCOPE — user asks to DO something / beyond chatbot’s scope
      - PROGRAMMING_HELP — asks for code/debugging/instructions
      - UNKNOWN_CAPABILITY — asks if Keena knows X that isn’t in profile
      - UNKNOWN_PREFERENCE — asks about likes/preferences not in profile
      - OFF_TOPIC — not about Keena/persona
      - ERROR — judge failed; default to ALLOW upstream
    """
    refusals = (policy or {}).get("refusal_messages", {})
    msg_unknown_cap = refusals.get("unknown_capability", "I don’t have that in my profile.")
    msg_unknown_pref = refusals.get("unknown_preference", "I don’t have info on that preference.")
    msg_outside = refusals.get("capability_outside", "I can’t do that. I only answer based on the info in my profile.")
    msg_code = refusals.get("code_help", "I don’t provide programming help or code fixes. I’m happy to talk about my background, projects, or preferences instead.")

    caps = ", ".join(sorted(capabilities)) or "(none)"
    skills_list = sorted(list((profile_terms or {}).get("skills", set())))
    prefs_list = sorted(list((profile_terms or {}).get("preferences", set())))

    lexicon_json = json.dumps({
        "skills": skills_list,
        "preferences": prefs_list,
    }, ensure_ascii=False)

    system_prompt = (
        "You are a policy checker for a personal chatbot (KeenaBot).\n"
        "Classify the USER intent and decide whether to send the assistant DRAFT as-is.\n"
        "If a skill/preference is asked and not in the provided LEXICON, return an UNKNOWN_* decision.\n"
        "Output ONLY JSON with keys: decision, reason, missing (array of strings), suggest_reply.\n"
        "Decisions: ALLOW | OUT_OF_SCOPE | PROGRAMMING_HELP | UNKNOWN_CAPABILITY | UNKNOWN_PREFERENCE | OFF_TOPIC.\n"
        f"Allowed capabilities: {caps}.\n"
        "Treat requests to write/fix/debug/generate code as PROGRAMMING_HELP.\n"
        "Treat requests to perform actions or tasks as OUT_OF_SCOPE.\n"
    )

    user_payload = {
        "LEXICON": json.loads(lexicon_json),
        "USER": user_text or "",
        "ASSISTANT_DRAFT": draft_answer or "",
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return "ERROR", f"judge_error: {e}", ""

    try:
        block = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(block.group(0) if block else raw)
        decision = str(parsed.get("decision", "ALLOW")).upper()
        reason = str(parsed.get("reason", ""))
        missing = parsed.get("missing", []) or []
        suggest = str(parsed.get("suggest_reply", ""))
    except Exception:
        # Very defensive fallback
        decision = "ALLOW" if "ALLOW" in raw.upper() else "REFUSE"
        reason = raw
        missing = []
        suggest = ""

    # Provide default suggestions if judge didn’t include one
    if decision == "UNKNOWN_CAPABILITY" and not suggest:
        if missing:
            suggest = f"I don’t have that in my profile: {', '.join(missing)}."
        else:
            suggest = msg_unknown_cap
    if decision == "UNKNOWN_PREFERENCE" and not suggest:
        if missing:
            suggest = f"I don’t have info on that preference in my profile: {', '.join(missing)}."
        else:
            suggest = msg_unknown_pref
    if decision == "OUT_OF_SCOPE" and not suggest:
        suggest = msg_outside
    if decision == "PROGRAMMING_HELP" and not suggest:
        suggest = msg_code

    return decision, reason, suggest

def build_guardrails(_capabilities: Set[str], _policy: Dict):
    """Kept for compatibility; returns a guard that never blocks pre-call."""
    def guard(_user_text: str):
        return False, ""
    return guard
