"""
ResearchMind AI
================
A Retrieval-Augmented Generation (RAG) research assistant built with
Streamlit, LangChain, FAISS, and Groq (Llama 3.3 70B).

Author: ResearchMind AI Team
"""

import os
import subprocess
import sys
import streamlit as st

from rag import ResearchAgent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR = "documents"
VECTOR_STORE_PATH = "vector_store/index.faiss"
EMBEDDING_MODEL_NAME = "MiniLM-L6-v2"
LLM_NAME = "Llama 3.3 70B"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all keys used in st.session_state exactly once."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session_state()


# ---------------------------------------------------------------------------
# Agent loading (cached so it is not rebuilt on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading research agent...")
def load_agent() -> ResearchAgent:
    return ResearchAgent()


agent = load_agent()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def count_pdfs(directory: str) -> int:
    """Return the number of PDF files in a directory (0 if missing)."""
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])


def save_uploaded_pdfs(files) -> int:
    """Persist uploaded PDFs to DOCS_DIR and return how many were saved."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = 0
    for pdf in files:
        save_path = os.path.join(DOCS_DIR, pdf.name)
        with open(save_path, "wb") as f:
            f.write(pdf.getbuffer())
        saved += 1
    return saved


def rebuild_vector_store() -> None:
    """Run the vector-store build script and refresh the cached agent."""
    subprocess.run(
    [sys.executable, "build_vector_store.py"],
    check=True
)
    load_agent.clear()
    st.session_state.agent_reloaded = True


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔬 ResearchMind AI")
st.caption("Advanced Research Agent | FAISS | Groq | LangChain | Citations")
st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.divider()

    # --- Upload PDFs -------------------------------------------------------
    st.subheader("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("💾 Save PDFs"):
        saved_count = save_uploaded_pdfs(uploaded_files)
        st.success(f"{saved_count} PDF(s) saved successfully.")

    st.divider()

    # --- Rebuild knowledge base ---------------------------------------------
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Building vector store..."):
            try:
                rebuild_vector_store()
                st.success("Knowledge base updated!")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to build vector store: {e}")

    st.divider()

    # --- Statistics ----------------------------------------------------------
    st.subheader("📊 Statistics")

    paper_count = count_pdfs(DOCS_DIR)
    vector_ready = "✅ Ready" if os.path.exists(VECTOR_STORE_PATH) else "❌ Not Built"

    st.metric("Research Papers", paper_count)
    st.metric("Vector Store", vector_ready)
    st.metric("Embedding Model", EMBEDDING_MODEL_NAME)
    st.metric("LLM", LLM_NAME)

    st.divider()

    # --- Pipeline overview -----------------------------------------------
    st.subheader("🔄 Pipeline")
    st.markdown(
        """
Question
⬇️
Embedding
⬇️
FAISS Search
⬇️
MMR Retrieval
⬇️
Groq LLM
⬇️
Answer + Citations
"""
    )

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat input + response generation
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a research question...")

if question:
    # Show and store the user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer
    with st.spinner("🔍 Searching vector database and generating answer..."):
        try:
            result = agent.answer(question)
        except Exception as e:
            result = {"answer": f"⚠️ An error occurred: {e}", "citations": []}

    # Display the assistant response
    with st.chat_message("assistant"):
        st.markdown(result["answer"])

        citations = result.get("citations", [])
        if citations:
            st.markdown("---")
            st.markdown("## 📚 Sources Used")
            for citation in citations:
                pages = ", ".join(str(page) for page in citation.get("pages", []))
                with st.expander(f"📄 {citation['file']}"):
                    st.write(f"**Pages:** {pages if pages else 'N/A'}")
        else:
            st.info("No supporting research documents were found.")

    # Persist the assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )


# ---------------------------------------------------------------------------
# Footer / project information
# ---------------------------------------------------------------------------

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
### 📚 Knowledge Base
✔ Research Papers
✔ Semantic Search
✔ Citations
"""
    )

with col2:
    st.success(
        """
### ⚡ AI Stack
• LangChain
• FAISS
• Groq
• Streamlit
"""
    )

with col3:
    st.warning(
        f"""
### 🧠 LLM
Model
{LLM_NAME}

Embedding
{EMBEDDING_MODEL_NAME}
"""
    )

st.divider()

st.caption(
    """
🔬 **ResearchMind AI**

Built using **Streamlit**, **LangChain**, **FAISS**, **Groq Llama 3.3**, and **Sentence Transformers**.

This research agent retrieves relevant passages from research papers, generates grounded answers,
provides citations, and clearly indicates when the supplied documents do not contain sufficient information.
"""
)