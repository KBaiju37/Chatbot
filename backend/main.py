import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("GROQ KEY:", os.getenv("GROQ_API_KEY"))
print("GEMINI KEY:", os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    message: str


# -------- SHARED COMPANY CONTEXT --------
COMPANY_SYSTEM_PROMPT = """
You are an AI assistant for Cerebrospark Innovations.

Company Overview:
Cerebrospark Innovations is a Pune-based drone manufacturing and drone solutions company.
We design, build, and deploy advanced drones and provide end-to-end drone solutions.

What We Do:
- Drone manufacturing (80g to 100kg payload capacity)
- Custom drone solutions for businesses
- End-to-end solutions from design to deployment

Industries:
- Agriculture
- Security & surveillance
- Delivery & logistics
- Healthcare
- AI-powered drone applications

Expertise:
- High-quality manufacturing
- AI & analytics integration
- Custom hardware + software systems

Solutions:
- Tailored drone configurations
- Software integration
- Data analytics & AI insights

Commitment:
- Industry-standard manufacturing
- Durable, reliable drones
- Consistent real-world performance

Technology:
- Continuous R&D
- AI-enabled drone systems
- Future-ready innovation

Client Approach:
- Customized solutions
- Long-term partnerships
- Full lifecycle support

Leadership:
- Mr. Ganesh Thorat (CEO)
- Mr. Mihir Kedar (CMO & CTO)
- Mr. Rushikesh Sonawane (COO)

Rules:
- Answer ONLY based on this company
- Do NOT invent unrelated industries
- Be professional, clear, and concise
- If unsure, ask for clarification instead of guessing
"""


# -------- GROQ FUNCTION --------
def ask_groq(message: str):
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": COMPANY_SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ]
            },
            timeout=30
        )

        print("GROQ STATUS:", res.status_code)
        print("GROQ RAW:", res.text)

        if res.status_code != 200:
            return None

        data = res.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        return None

    except Exception as e:
        print("GROQ ERROR:", e)
        return None


# -------- GEMINI FUNCTION --------
def ask_gemini(message: str):
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"

        res = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"""
{COMPANY_SYSTEM_PROMPT}

User Question:
{message}

Answer professionally based on the company.
"""
                            }
                        ]
                    }
                ]
            },
            timeout=30
        )

        print("GEMINI STATUS:", res.status_code)
        print("GEMINI RAW:", res.text)

        if res.status_code != 200:
            return None

        data = res.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            parts = data["candidates"][0]["content"]["parts"]
            if parts and "text" in parts[0]:
                return parts[0]["text"]

        return None

    except Exception as e:
        print("GEMINI ERROR:", e)
        return None


# -------- ROUTE --------
@app.post("/chat")
async def chat(req: ChatRequest):
    message = req.message.strip()

    # 1) Try Groq
    reply = ask_groq(message)
    model = "groq"

    # 2) Fallback to Gemini
    if not reply or len(reply.strip()) == 0:
        reply = ask_gemini(message)
        model = "gemini"

    # 3) Final fallback
    if not reply:
        reply = "Sorry, I’m unable to respond right now. Please try again shortly."

    return {
    "reply": reply,
    "model": model,
    "success": True
}