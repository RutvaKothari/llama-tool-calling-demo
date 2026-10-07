import requests
import json
from word2number import w2n

def add_numbers(a, b):
    return a + b

def parse_number(num_str):
    """
    Attempts to parse a string into a number.
    First tries direct integer conversion (for '459', '2000').
    If that fails, tries word2number (for 'five', 'forty two thousand').
    """
    if not num_str:
        return None
        
    # Clean up the string a bit
    num_str = str(num_str).strip().replace(',', '')
    
    # Try direct integer conversion first
    try:
        return int(num_str)
    except ValueError:
        pass
        
    # Try word2number conversion
    try:
        return w2n.word_to_num(num_str)
    except ValueError:
        return None

def analyze_input(user_input):
    """
    Uses the LLM ONLY to classify the intent and extract the raw number strings.
    It does NOT perform any math or conversion.
    """
    system_prompt = (
    "You are an intent classifier and entity extractor.\n\n"

    "Your ONLY job is to determine whether the user is EXPLICITLY asking you to perform an addition calculation.\n"

    "A request should be classified as an addition request ONLY if the user clearly expresses an intent to calculate a sum.\n"

    "Never infer addition simply because two or more numbers appear in the sentence.\n"

    "Numbers may represent quantities, ages, dates, prices, phone numbers, addresses, scores, measurements, IDs, inventory, or any other information. These are NOT addition requests unless the user explicitly asks to calculate.\n"

    "A sentence is an addition request ONLY if it contains clear mathematical intent using words or symbols such as:\n"
    "\"add\", \"plus\", \"+\", \"sum\", \"total\", \"calculate\", \"what is\", \"how much is\", \"find the sum\", \"combine\", or any other clear request to perform addition.\n"

    "If the user is merely describing something, reporting facts, comparing quantities, listing items, or telling a story, classify it as NOT an addition request even if multiple numbers are present.\n"

    "Examples of VALID addition requests:\n"
    "- '34930 + 4894'\n"
    "- 'add 3493 and 4894'\n"
    "- 'What is 200 plus 500?'\n"
    "- 'Calculate the sum of 15 and 25.'\n"
    "- 'Can you add five hundred and forty two thousand?'\n"

    "Examples of NOT addition requests:\n"
    "- 'I bought 15 apples and 20 oranges today.'\n"
    "- 'I have 2 cats and 3 dogs.'\n"
    "- 'My salary is 50000 and my bonus is 10000.'\n"
    "- 'The score was 180 for India and 175 for Australia.'\n"
    "- 'I am 20 years old and my brother is 18.'\n"
    "- 'There are 12 chairs and 5 tables in the room.'\n"
    "- 'The meeting starts at 3 PM on 25 July.'\n"
    "- 'My phone number is 9876543210.'\n"
    "- 'Room 201 has 4 beds.'\n"
    "- 'I ate 3 pizzas and drank 2 bottles of water.'\n"

    "Extract the two numbers exactly as they appear in the user's text without converting them.\n"

    "Respond ONLY with valid JSON in exactly this format:\n"

    "{\n"
    "  \"is_addition_request\": true/false,\n"
    "  \"num1_raw\": \"string or null\",\n"
    "  \"num2_raw\": \"string or null\"\n"
    "}\n"

    "If there is ANY doubt about the user's intent, return:\n"

    "{\n"
    "  \"is_addition_request\": false,\n"
    "  \"num1_raw\": null,\n"
    "  \"num2_raw\": null\n"
    "}\n"

    "Be conservative. It is much better to classify a request as NOT an addition request than to incorrectly perform a calculation."
)

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "format": "json",
            "stream": False
        }
     )

    raw_text = response.json()["message"]["content"]

    try:
        parsed = json.loads(raw_text)
        return parsed
    except (json.JSONDecodeError, KeyError):
        return {"is_addition_request": False, "num1_raw": None, "num2_raw": None}

def normal_chat(messages_for_chat):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": messages_for_chat,
            "stream": False
        }
     )
    return response.json()["message"]["content"]

# ---- MAIN LOOP ----

conversational_messages = [
    {"role": "system", "content": "You are a friendly, natural conversational assistant."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Step 1: LLM analyzes intent and extracts raw strings
    analysis = analyze_input(user_input)
    # print("DEBUG analysis:", analysis)

    if analysis.get("is_addition_request"):
        num1_raw = analysis.get("num1_raw")
        num2_raw = analysis.get("num2_raw")

        # Step 2: Python parses the raw strings into integers
        num1 = parse_number(num1_raw)
        num2 = parse_number(num2_raw)

        if num1 is not None and num2 is not None:
            # Step 3: Python performs the addition
            result = add_numbers(num1, num2)
            print("AI:", f"The result is {result}.")
            # Note: We do NOT add math interactions to conversational_messages
        else:
            # If parsing failed (e.g., word2number couldn't handle it), fall back to chat
            print("DEBUG: Failed to parse numbers. Falling back to chat.")
            conversational_messages.append({"role": "user", "content": user_input})
            reply = normal_chat(conversational_messages)
            print("AI:", reply)
            conversational_messages.append({"role": "assistant", "content": reply})
    else:
        # Step 4: Non-addition requests go to normal chat
        conversational_messages.append({"role": "user", "content": user_input})
        reply = normal_chat(conversational_messages)
        print("AI:", reply)
        conversational_messages.append({"role": "assistant", "content": reply})