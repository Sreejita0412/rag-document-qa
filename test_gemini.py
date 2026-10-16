

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
else:
    print("API key loaded successfully!")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key
    )

    response = llm.invoke(
        "Reply with only: Gemini connection successful!"
    )

    print(response.content)
