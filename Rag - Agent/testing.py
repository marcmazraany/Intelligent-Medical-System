from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="./chroma_text_store",
    embedding_function=embeddings
)

collection = vectordb._collection
print(f"Total chunks in database: {collection.count()}")

results = vectordb.similarity_search("medication", k=3)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:200]}...")