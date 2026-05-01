from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
app = FastAPI(title="IntelliMeds AI Bridge")


# -----------------------------
# Config
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_text_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)


# -----------------------------
# DTOs
# -----------------------------
class MedicationItem(BaseModel):
    name: str
    dosage: Optional[str] = None
    quantity: Optional[int] = None
    frequency: Optional[str] = None


class HistoryItem(BaseModel):
    role: str
    content: str


class AnalyzeRequest(BaseModel):
    symptoms: str = Field(..., min_length=1)
    medications: List[MedicationItem] = []
    history: List[HistoryItem] = []


class AnalyzeResponse(BaseModel):
    reply: str


# -----------------------------
# Helpers
# -----------------------------
def retrieve_medical_context(symptoms: str, k: int = 5) -> str:
    docs = vector_store.similarity_search(symptoms, k=k)
    if not docs:
        return ""
    return "\n\n".join([doc.page_content for doc in docs])


def format_medications(medications: List[MedicationItem]) -> str:
    if not medications:
        return "No user medications available."

    lines = []
    for med in medications:
        parts = [f"name: {med.name}"]
        if med.dosage:
            parts.append(f"dosage: {med.dosage}")
        if med.frequency:
            parts.append(f"frequency: {med.frequency}")
        if med.quantity is not None:
            parts.append(f"quantity: {med.quantity}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def format_history(history: List[HistoryItem]) -> str:
    if not history:
        return "No previous conversation history."

    return "\n".join([f"{item.role}: {item.content}" for item in history[-6:]])


def extract_med_names(medications: List[MedicationItem]) -> List[str]:
    return [m.name for m in medications if m.name and m.name.strip()]


def medication_mentions_in_reply(reply: str, med_names: List[str]) -> List[str]:
    lowered = reply.lower()
    mentioned = []
    for name in med_names:
        if name.lower() in lowered:
            mentioned.append(name)
    return mentioned


# -----------------------------
# Core endpoint
# -----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        retrieved_context = retrieve_medical_context(request.symptoms, k=5)
        medication_text = format_medications(request.medications)
        history_text = format_history(request.history)
        available_med_names = extract_med_names(request.medications)

        prompt = f"""
You are a cautious medical guidance assistant for an educational medication-management app.

Your task:
1. Consider the user's current symptoms.
2. Use the retrieved medical context if relevant.
3. Consider the user's available medications.
4. Consider the recent conversation history.
5. Give a concise, practical response.

Rules:
- Do not claim a definitive diagnosis.
- Use phrases like "could be", "may be", or "might indicate".
- Do not provide dangerous instructions.
- Do not provide prescription-only recommendations as if certain.
- Prefer general supportive advice when appropriate.
- Mention when the user should seek urgent medical attention.
- If any of the user's available medications are relevant, mention them cautiously.
- Keep the answer readable and not too long.

User symptoms:
{request.symptoms}

Recent conversation history:
{history_text}

User available medications:
{medication_text}

Retrieved medical context:
{retrieved_context}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful medical information assistant. Provide cautious, non-diagnostic, concise guidance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        reply = response.choices[0].message.content.strip()

        mentioned_user_meds = medication_mentions_in_reply(reply, available_med_names)
        if mentioned_user_meds:
            reply += "\n\nRelevant medications from your inventory that were mentioned: " + ", ".join(mentioned_user_meds)

        reply += "\n\nThis is general information only and not a medical diagnosis. Seek professional care if symptoms are severe, worsening, or unusual."

        return AnalyzeResponse(reply=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")