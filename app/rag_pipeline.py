
from dotenv import load_dotenv
import os
from pathlib import Path


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

DATA_DIR = Path("data")
FAISS_DIR = Path("embeddings/faiss_index")

def load_pdfs() -> list[Document]:
    documents = []
    for pdf in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def build_faiss_index(chunks: list[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_DIR))
    return vectorstore

def load_faiss_index() -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

def ingest_if_needed() -> FAISS:
    if FAISS_DIR.exists():
        return load_faiss_index()

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    return build_faiss_index(chunks)

def retrieve_context(query: str, k: int = 4) -> str:
    vectorstore = ingest_if_needed()
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
