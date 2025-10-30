# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# # Load .env
# load_dotenv()

# # # Get API key
# # api_key = os.environ.get("OPENAI_API_KEY")
# # if not api_key:
# #     raise ValueError("OPENAI_API_KEY not found in .env")
# # # Initialize client
# # client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# print("Client created successfully")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )
import os
from dotenv import load_dotenv
from google import genai
# from google.genai import types

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# -----------------------------
# Initialize Gemini client
# -----------------------------
client = genai.Client(api_key=api_key)
print("Gemini client created successfully")

# -----------------------------
# Prepare prompt for chat
# -----------------------------
prompt = "You are a helpful assistant. Say hello to the user."

# Optional: add tools or grounding (if needed)
# config = types.GenerateContentConfig()

# -----------------------------
# Generate response
# -----------------------------
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # or another Gemini model
        contents=prompt,
        # config=config
    )
    print("Gemini response:", response.text)
except Exception as e:
    print("Error calling Gemini:", str(e))

# print(response.choices[0].message.content)