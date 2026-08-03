import streamlit as st
from rag import ResearchAgent

st.set_page_config(
    page_title="Research Agent with Citations",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Agent with Citations")

st.markdown("""
Ask questions about the research papers in the knowledge base.

The agent retrieves relevant documents and answers using only the retrieved context.
""")

@st.cache_resource
def load_agent():
    return ResearchAgent()

agent = load_agent()

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Retrieval-Augmented Generation?"
)

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching research papers..."):

        result = agent.answer(question)

    st.subheader("Answer")

    st.write(result["answer"])

    st.subheader("Sources")

    if len(result["citations"]) == 0:

        st.info("No relevant sources found.")

    else:

        for citation in result["citations"]:

            pages = ", ".join(
                str(page)
                for page in citation["pages"]
            )

            st.success(
                f"""
📄 {citation['file']}

Pages: {pages if pages else "N/A"}
"""
            )