from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_embeddings

DATA_DIR = Path("data")
INDEX_DIR = "index"

def load_docs():
    docs = []
    for f in DATA_DIR.glob("*.txt"):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    return docs

def build_index():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents -> {INDEX_DIR}/")

if __name__ == "__main__":
    build_index()
