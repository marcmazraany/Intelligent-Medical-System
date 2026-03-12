import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import re

# ===================== LOAD ENV =====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== USER MED INVENTORY =====================
USER_MEDS = ["Ibuprofen", "Paracetamol", "Vitamin C"]

# ===================== RAG FUNCTION =====================
def medical_rag(symptoms):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="./Rag - Agent/chroma_text_store",
        embedding_function=embeddings
    )

    results = vectordb.similarity_search(symptoms, k=5)

    combined_text = "\n\n---\n\n".join(
        [f"Source: {r.metadata.get('source', 'unknown')} (page {r.metadata.get('page', '?')})\n{r.page_content}" 
         for r in results]
    )

    prompt = f"""
You are a helpful medical assistant.
Use the provided context to list possible conditions and safe home treatments.
Output in JSON with fields: condition, recommended_drugs, non_drug_measures.

User Symptoms: {symptoms}

Context:
{combined_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=400
    )

    return response.choices[0].message.content, combined_text

# ===================== MED MATCHER =====================
def medication_tool(advice):
    found = [m for m in USER_MEDS if re.search(m, advice, re.IGNORECASE)]
    if found:
        return f"Based on the medications available in your drawer {USER_MEDS} You can use: {', '.join(found)}"
    return "⚠️ None of your stored medications are directly recommended."

# ===================== ORCHESTRATOR =====================
def orchestrator(symptoms, rag_output):
    prompt = f"""
You are a medical orchestrator.
User symptoms: {symptoms}
RAG output: {rag_output}

Generate safe, short medical advice.
Mention only medications that are appropriate.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You summarize safely and conservatively."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )

    return response.choices[0].message.content

# ===================== STREAMLIT UI =====================

st.set_page_config(page_title="AI Medical Assistant", layout="centered")

st.title("🧠 AI Medical Assistant (RAG + Medication Awareness)")

st.markdown("### Enter Patient Symptoms")
symptoms = st.text_area("Example: sore throat, fever, headache")

st.markdown("---")
st.markdown("### Patient Medication Inventory")
st.write(", ".join(USER_MEDS))

st.markdown("---")

if st.button("Run Medical AI"):
    if not symptoms:
        st.warning("Please enter symptoms.")
    else:
        with st.spinner("Analyzing using medical knowledge base..."):
            rag_output, context = medical_rag(symptoms)

        with st.spinner("Generating safe advice..."):
            final_advice = orchestrator(symptoms, rag_output)

        med_result = medication_tool(final_advice)

        # ===================== DISPLAY =====================

        st.success("✅ Analysis Complete")

        with st.expander("📚 Retrieved Medical Context (RAG Sources)"):
            st.text(context)

        with st.expander("🧬 RAG Structured Output"):
            st.text(rag_output)

        st.markdown("## 🧾 Final Medical Advice")
        st.info(final_advice)

        st.markdown("## 💊 Medication Match")
        st.success(med_result)

        st.markdown("---")
        st.caption("⚠️ This system provides educational assistance only and does not replace professional medical care.")
