from Utilities.loader import load_documents
from Utilities.splitter import split_documents

docs = load_documents("data/raw_docs")
print(f"Loaded {len(docs)} documents")

chunks = split_documents(docs)
print(f"Split into {len(chunks)} chunks")

print(chunks[0].page_content[:500])
