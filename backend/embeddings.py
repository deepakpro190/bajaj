# backend/embeddings.py
import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ChromaDB client
client     = chromadb.PersistentClient(path="./vector_store")
collection = client.get_or_create_collection(name="insurance_docs")

# Local model for embeddings
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_local(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts).tolist()

def embed_and_store(doc_id: str, texts: list[str]):
    embs = embed_local(texts)
    # store in Chroma under the doc_id namespace
    for idx, (chunk, emb) in enumerate(zip(texts, embs)):
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[f"{doc_id}_{idx}"],
            metadatas=[{"doc_id": doc_id}],
        )
    return embs

def retrieve_relevant(query: str, k: int = 5) -> list[str]:
    q_emb = embed_local([query])[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    return results["documents"][0]

'''
import os
import shutil
from chromadb import PersistentClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# CONFIGURATION
PDF_FOLDER = "files"
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "insurance_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def reset_vector_store(path: str):
    """Deletes existing Chroma vector store"""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"✅ Cleared existing vector store at '{path}'")
        except Exception as e:
            print(f"❌ Failed to delete vector store: {e}")
            raise

def load_pdfs(folder_path):
    """Loads and returns documents from all PDFs in a folder"""
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            print(f"📄 Loading {file}...")
            loader = PyPDFLoader(full_path)
            pages = loader.load()
            all_docs.extend(pages)
    return all_docs

def split_documents(docs):
    """Splits documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    return splitter.split_documents(docs)

def embed_and_store_documents(docs, vector_store_path, collection_name):
    """Embeds documents using HuggingFace and stores in ChromaDB"""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    chroma_client = PersistentClient(path=vector_store_path)
    
    collection = chroma_client.get_or_create_collection(name=collection_name)
    vectorstore = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embedding)
    
    print(f"🔗 Storing {len(docs)} chunks in ChromaDB...")
    vectorstore.add_documents(docs)
    print("✅ Embedding and storage complete.")

def main():
    print("🚀 Starting embedding process...")

    reset_vector_store(VECTOR_STORE_PATH)

    raw_docs = load_pdfs(PDF_FOLDER)
    print(f"📄 Total pages loaded: {len(raw_docs)}")

    chunks = split_documents(raw_docs)
    print(f"✂️ Total chunks after splitting: {len(chunks)}")

    embed_and_store_documents(chunks, VECTOR_STORE_PATH, COLLECTION_NAME)

    print("✅ All documents embedded and stored successfully.")

if __name__ == "__main__":
    main()
'''