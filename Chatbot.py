import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from guardrails import build_guardrails, system_rules_text
from context_selector import select_context

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ABOUT_ME_FILE = "about_me.json"
with open(ABOUT_ME_FILE, "r", encoding="utf-8") as f:
    about_me_data = json.load(f)

CAPABILITIES = set(about_me_data.get("capabilities", []))
POLICY = about_me_data.get("policy", {})
# Build guard function
guardrails = build_guardrails(CAPABILITIES, POLICY)

#Opening message
print("Hello! I am an AI chatbot designed to respond as Keena would with a good amount of information about anything you could want to know. Feel free to ask anything about me, my life, or my work experience.")

#User Input part
def collect_user_input():
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit", "bye", "goodbye", "good bye"}:
            print("Goodbye!")
            break

        blocked, msg = guardrails(user_input)
        if blocked:
            print(f"KeenaBot: {msg}")
            continue

        try:
            reply = generate_response(user_input)
            print(f"KeenaBot: {reply}")
        except Exception as e:
            print(f"Error: {e}")

#Response Part
def generate_response(user_input):
    system_text = system_rules_text(CAPABILITIES)

    # Select only the relevant parts of the json for the query
    context = select_context(user_input, about_me_data)
    response = client.chat.completions.create(
        model="gpt-5",
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": (
                    "BACKGROUND (selected sections from about_me.json):\n" + context + "\n\n" +
                    "USER MESSAGE:\n" + user_input
                ),
            },
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    collect_user_input()
