import streamlit as st
from rag import ResearchAgent

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Agent")
st.caption(
    "Powered by FAISS • Groq Llama 3.3 • Semantic Search • Citations"
)
st.markdown("""
Ask questions about the research papers in the knowledge base.

The agent retrieves relevant documents and answers using only the retrieved context.
""")


class chat_message:
    """A lightweight wrapper around Streamlit's chat UI."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.content = content
        self._message = None

    def __enter__(self):
        self._message = st.chat_message(self.role)
        return self._message.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._message is not None:
            return self._message.__exit__(exc_type, exc_value, traceback)
        return False

    def render(self, content: str | None = None):
        with self:
            message = content if content is not None else self.content
            if message:
                st.markdown(message)


@st.cache_resource
def load_agent():
    return ResearchAgent()

agent = load_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
st.sidebar.title("🔧 Control Panel")

st.sidebar.header("📊 Project Statistics")

st.sidebar.metric("Research Papers", "99")
st.sidebar.metric("Pages", "2014")
st.sidebar.metric("Chunks", "8522")

st.sidebar.divider()

st.sidebar.write("### 🤖 Model")

st.sidebar.write("Embedding")
st.sidebar.code("all-MiniLM-L6-v2")

st.sidebar.write("LLM")
st.sidebar.code("Llama 3.3 70B")

st.sidebar.write("Vector Database")
st.sidebar.code("FAISS")
uploaded_files = st.sidebar.file_uploader(
    "📂 Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} PDF(s) uploaded."
    )

    st.sidebar.info(
        "Save them into the documents folder and rebuild the vector store."
    )

for message in st.session_state.messages:
    chat_message(message["role"], message["content"]).render()

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Retrieval-Augmented Generation?"
)

if st.button("Ask"):
    st.session_state.messages.append(
    {
        "role": "user",
        "content": question,
    }
)

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching research papers..."):

        result = agent.answer(question)
        st.session_state.messages.append(
    {
        "role": "assistant",
        "content": result["answer"],
    }
)

    st.subheader("Answer")

    st.info(result["answer"])

    st.subheader("Sources")

    if len(result["citations"]) == 0:
        st.info("No relevant sources found.")
    else:
        for citation in result["citations"]:
            pages = ", ".join(str(page) for page in citation["pages"])

            with st.expander(f"📄 {citation['file']}"):
                st.write(f"**Pages:** {pages if pages else 'N/A'}")
                st.divider()

st.caption(
    "Developed using FAISS, Groq, LangChain and Streamlit"
)