from langchain.document_loaders import PyPDFLoader
import os

def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents