from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Optional, List
import re

from queries import (
    get_user_medications,
    get_conversation_history,
    save_message
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== GLOBAL INITIALIZATION =====================

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTORDB = Chroma(persist_directory="chroma_text_store", embedding_function=EMBEDDINGS)


class GrapheState(TypedDict):
    symptoms: str
    user_id: str
    conversation_id: str
    history: Optional[List]
    rag_output: Optional[dict]
    final_advice: Optional[str]


# ===================== MEMORY NODE =====================

def memory_node(state):

    conversation_id = state["conversation_id"]

    history = get_conversation_history(conversation_id)

    # limit history to last 6 messages (3 exchanges)
    history = history[-6:]

    return {"history": history}


# ===================== RAG NODE =====================

def medical_rag_node(state):

    query = state["symptoms"]
    history = state.get("history", [])

    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history]
    )

    k = 5

    results = VECTORDB.similarity_search(query, k=k)

    combined_text = "\n\n---\n\n".join(
        [
            f"Source: {r.metadata.get('source', 'unknown')} (page {r.metadata.get('page', '?')})\n{r.page_content}"
            for r in results
        ]
    )

    prompt = f"""
You are a helpful medical assistant.

Conversation history:
{history_text}

Use the provided context to list possible conditions and safe home treatments.

Output in JSON with fields:
condition, recommended_drugs, non_drug_measures.

User Symptoms:
{query}

Context:
{combined_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise, expert medical assistant. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=400
    )

    try:
        import json
        rag_output = json.loads(response.choices[0].message.content)
    except Exception:
        rag_output = {
            "error": "Failed to parse AI response",
            "raw": response.choices[0].message.content
        }

    return {"rag_output": rag_output}


# ===================== MEDICATION TOOL =====================

def medication_tool(advice_text, user_meds):

    meds_in_advice = [
        m for m in user_meds
        if re.search(m, advice_text, re.IGNORECASE)
    ]

    if meds_in_advice:
        return f"Based on your available medications ({', '.join(user_meds)}), you can take: {', '.join(meds_in_advice)}."
    else:
        return "None of your available medications are mentioned in the recommendations."


# ===================== ORCHESTRATOR =====================

def orchestrator_node(state):

    rag_output = state["rag_output"]
    symptoms = state["symptoms"]
    user_id = state["user_id"]
    conversation_id = state["conversation_id"]

    user_meds = get_user_medications(user_id)

    prompt = f"""
You are a medical orchestrator.

The user reports these symptoms:
{symptoms}

The RAG agent returned:
{rag_output}

The user currently has these medications:
{user_meds}

Summarize safe, concise medical advice.
Mention medications that could help.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a clear, safe medical summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )

    llm_advice = response.choices[0].message.content

    matched_meds = medication_tool(llm_advice, user_meds)

    final = f"{llm_advice}\n\n{matched_meds}"

    # ================= SAVE MEMORY =================

    save_message(conversation_id, "user", symptoms)
    save_message(conversation_id, "assistant", final)

    return {"final_advice": final}


# ===================== GRAPH =====================

graph = StateGraph(GrapheState)

graph.add_node("memory", memory_node)
graph.add_node("medical_rag", medical_rag_node)
graph.add_node("orchestrator", orchestrator_node)

graph.add_edge("memory", "medical_rag")
graph.add_edge("medical_rag", "orchestrator")

graph.set_entry_point("memory")
graph.set_finish_point("orchestrator")

app = graph.compile()


# ===================== TEST =====================

if __name__ == "__main__":

    result = app.invoke({
        "symptoms": "I have a sore throat, mild fever, and headache",
        "user_id": "123",
        "conversation_id": "conv_1"
    })

    print("\n=== FINAL MEDICAL ADVICE ===\n")
    print(result["final_advice"])