import os
from langchain_google_genai import ChatGoogleGenerativeAI

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Set GEMINI_API_KEY before running this file.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = llm.invoke("Write a short story about a brave knight.")
print(result.content)
