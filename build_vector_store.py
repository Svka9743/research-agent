from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("=" * 60)
print("STEP 1 : Loading PDFs")
print("=" * 60)

loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

print("=" * 60)
print("STEP 2 : Splitting Documents")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("=" * 60)
print("STEP 3 : Loading Embedding Model")
print("=" * 60)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("=" * 60)
print("STEP 4 : Building FAISS")
print("=" * 60)

db = FAISS.from_documents(
    chunks,
    embeddings
)

Path("vector_store").mkdir(exist_ok=True)

db.save_local("vector_store")

print("=" * 60)
print("SUCCESS")
print("=" * 60)
print("Vector Store Saved")