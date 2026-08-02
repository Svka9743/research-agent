import os
from dotenv import load_dotenv

load_dotenv()

# ===========================
# API Keys
# ===========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===========================
# Paths
# ===========================

VECTOR_STORE_PATH = "vector_store"

# ===========================
# Embedding Model
# ===========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ===========================
# Retrieval Settings
# ===========================

TOP_K = 5

# ===========================
# Groq Model
# ===========================

LLM_MODEL = "llama-3.3-70b-versatile"