from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------- Load environment variables ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(" Missing OpenAI API key. Add OPENAI_API_KEY=your_key to .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- 1. Extract text from PDFs ----------
def extract_text_from_pdf(pdf_path):
    """Extract readable text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "content": text})
    return pages


# ---------- 2. Chunking ----------
def chunk_text_pages(pages, chunk_size=800, chunk_overlap=150):
    """Split PDF pages into manageable text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for p in pages:
        for chunk in text_splitter.split_text(p["content"]):
            chunks.append(Document(page_content=chunk, metadata={"page": p["page"]}))
    return chunks


# ---------- 3. Process all PDFs ----------
def process_all_pdfs(folder_path="data"):
    """Load and process all PDFs in the given folder."""
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            print(f"[PDF] Processing: {filename}")
            pdf_path = os.path.join(folder_path, filename)
            pages = extract_text_from_pdf(pdf_path)
            chunks = chunk_text_pages(pages)
            for c in chunks:
                c.metadata["source"] = filename
            all_chunks.extend(chunks)
    return all_chunks


# ---------- 4. Store chunks in Chroma ----------
def store_in_vector_db(chunks, persist_dir="chroma_text_store"):
    """Store text chunks into a persistent vector database."""
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print(f" Stored {len(chunks)} chunks in Chroma at '{persist_dir}'")
    return vectordb


# ---------- 5. Analyze chunks using OpenAI ----------
def analyze_chunks_with_llm(chunks, top_n=5):
    """Send a few chunks to OpenAI LLM for reasoning/summarization."""
    print(f"\n [LLM] Analyzing {len(chunks)} chunks...")

    subset = chunks[:top_n]
    combined_text = "\n\n---\n\n".join([
        f"Source: {c.metadata['source']} (Page {c.metadata['page']})\n{c.page_content}"
        for c in subset
    ])

    prompt = f"""
You are a concise and knowledgeable assistant.
Analyze the following text chunks and provide:
1. A short summary of the most important points.
2. Identify which chunks (by source and page) are most informative.

Text:
{combined_text}

Output format:
Summary:
[Your concise summary here]

Most informative chunks:
[List source and page numbers]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can switch to "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": "You are an expert summarizer and analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=400
        )

        answer = response.choices[0].message.content
        print("\n [OpenAI Response]:\n")
        print(answer)

    except Exception as e:
        print(f"\n OpenAI API Error: {str(e)}")


# ---------- 6. Run everything ----------
if __name__ == "__main__":
    chunks = process_all_pdfs("data")
    vectordb = store_in_vector_db(chunks)
    analyze_chunks_with_llm(chunks)
