import json

def select_context(user_input: str, data: dict) -> str:
    text = (user_input or "").lower()
    sections = []

    if any(k in text for k in ["skill", "stack", "tool", "tech", "technology"]):
        sections.append({"skills": data.get("skills", {})})

    if any(k in text for k in ["job", "work", "experience", "role", "company", "employment", "project"]):
        sections.append({"experience": data.get("experience", [])})

    if any(k in text for k in ["school", "education", "degree", "wgu", "university", "college"]):
        sections.append({"education": data.get("education", [])})

    if any(k in text for k in ["cert", "license", "certification", "certifications"]):
        sections.append({"certifications": data.get("certifications", [])})

    if any(k in text for k in ["personal", "hobby", "hobbies", "favorite", "favorites", "kids", "children", "family", "color", "food", "foods"]):
        sections.append({"personal": data.get("personal", {})})

    if any(k in text for k in ["kill", "suicide", "unalive", "hurt"]):
        sections.append({"suicide": data.get("suicide", [])})

    if any(k in text for k in ["contact", "email", "phone", "address", "website", "linkedin"]):
        if data.get("contact"):
            sections.append({"contact": data.get("contact")})
        if data.get("websites"):
            sections.append({"websites": data.get("websites")})

    if not sections:
        sections.append({
            "persona": {
                "full_name": data.get("persona", {}).get("full_name"),
                "headline": data.get("persona", {}).get("headline"),
                "summary": data.get("persona", {}).get("summary"),
            }
        })

    return json.dumps(sections, ensure_ascii=False)