import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config import (
    GROQ_API_KEY,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    LLM_MODEL,
)

from prompts import SYSTEM_PROMPT

load_dotenv()


class ResearchAgent:

    def __init__(self):

        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env")

        print("Loading Embedding Model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Loading FAISS Vector Store...")

        self.vector_db = FAISS.load_local(
            VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        print("Connecting to Groq...")

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    """
Question:

{question}

Context:

{context}
""",
                ),
            ]
        )

    def retrieve(self, question):

        docs = self.vector_db.similarity_search_with_score(
            question,
            k=TOP_K,
        )

        return docs

    def build_context(self, docs):

        context = ""

        citations = []

        for document, score in docs:

            filename = os.path.basename(
                document.metadata.get("source", "Unknown")
            )

            citations.append(
                {
                    "file": filename,
                    "score": round(float(score), 4),
                }
            )

            context += (
                f"\n\nSOURCE: {filename}\n"
                f"{document.page_content}\n"
            )

        return context, citations

    def answer(self, question):

        docs = self.retrieve(question)

        if len(docs) == 0:

            return {
                "answer": "The provided source documents do not contain enough information to answer this question.",
                "citations": [],
            }

        context, citations = self.build_context(docs)

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return {
            "answer": response.content,
            "citations": citations,
        }


if __name__ == "__main__":

    agent = ResearchAgent()

    while True:

        question = input("\nAsk Question (type exit to quit): ")

        if question.lower() == "exit":
            break

        result = agent.answer(question)

        print("\n")
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result["answer"])

        print("\n")
        print("=" * 80)
        print("CITATIONS")
        print("=" * 80)

        for citation in result["citations"]:
            print(
                f"{citation['file']} | similarity score: {citation['score']}"
            )