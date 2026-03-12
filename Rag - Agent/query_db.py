from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def query_vectordb(query, k=5):
    """Search the vector database for relevant chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="./chroma_text_store",
        embedding_function=embeddings
    )

    results = vectordb.similarity_search(query, k=k)

    print(f"\n Query: '{query}' — found {len(results)} chunks\n")

    combined_text = "\n\n---\n\n".join(
        [f"Source: {r.metadata['source']} (page {r.metadata['page']})\n{r.page_content}" for r in results]
    )

    for i, doc in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Result #{i}")
        print(f" Source: {doc.metadata['source']}")
        print(f" Page: {doc.metadata['page']}")
        print(f"\n{doc.page_content[:400]}...\n")
        print(f"{'='*60}\n")

    analyze_with_openai(query, combined_text)



def analyze_with_openai(query, context_text):
    """Send relevant chunks to OpenAI for reasoning."""
    print("\n Sending to OpenAI GPT model for reasoning...\n")

    prompt = f"""
You are a helpful and knowledgeable assistant.
Use the provided context to clearly and accurately answer the question.
If the context doesn't contain enough information, say so briefly.

Question:
{query}

Context:
{context_text}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a concise, expert assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=350
        )

        answer = response.choices[0].message.content
        print("\n OpenAI Answer:\n")
        print(answer)

    except Exception as e:
        print(f"\n OpenAI Error: {str(e)}")

if __name__ == "__main__":
    query_vectordb("What should I do if I have a cold?", k=3)
