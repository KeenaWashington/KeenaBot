import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from guardrails import build_guardrails, system_rules_text
from context_selector import select_context

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABOUT_ME_FILE = os.path.join(BASE_DIR, "about_me.json")
try:
    with open(ABOUT_ME_FILE, "r", encoding="utf-8") as f:
        about_me_data = json.load(f)
except FileNotFoundError:
    about_me_data = {}
    print(f"Warning: {ABOUT_ME_FILE} not found; running with empty profile.")
except json.JSONDecodeError as e:
    about_me_data = {}
    print(f"Warning: Invalid JSON in {ABOUT_ME_FILE}: {e}")

CAPABILITIES = set(about_me_data.get("capabilities", []))
POLICY = about_me_data.get("policy", {})
# Build guard function
guardrails = build_guardrails(CAPABILITIES, POLICY)

# --- Conversation memory  ------------------------------------------------------
CHAT_HISTORY = []
MAX_TURNS = 6
# -------------------------------------------------------------------------------

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

    # Select only the relevant parts of the json for the current query
    context = select_context(user_input, about_me_data)

    # Build the messages array with a rolling window of history
    messages = [{"role": "system", "content": system_text}]
    #Chat History
    if CHAT_HISTORY:
        messages.extend(CHAT_HISTORY[-2 * MAX_TURNS:])
    messages.append({
        "role": "user",
        "content": (
            "BACKGROUND (selected sections from about_me.json):\n" + context + "\n\n" +
            "USER MESSAGE:\n" + user_input
        ),
    })

    response = client.chat.completions.create(
        model="gpt-5",
        reasoning_effort="low",
        messages=messages,
    )
    reply = response.choices[0].message.content

    #More chat history
    CHAT_HISTORY.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ])

    return reply

if __name__ == "__main__":
    print("Hello! I am an AI chatbot designed to respond as Keena would with a good amount of information about anything you could want to know. Feel free to ask anything about me, my life, or my work experience.")
    try:
        collect_user_input()
    except KeyboardInterrupt:
        print("\nGoodbye!")
