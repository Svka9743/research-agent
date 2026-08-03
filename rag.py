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
        retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

        documents = retriever.invoke(question)
        docs = []

        for doc in documents:
            score = doc.metadata.get("score")
            if score is None:
                score = 0.0
            docs.append((doc, float(score)))

        return docs

    def build_context(self, docs):

        context = ""

        citations = []

        seen = {}

        for document, score in docs:

            filename = os.path.basename(
                document.metadata.get("source", "Unknown")
            )

            page = document.metadata.get("page")
            if page is None:
                page = "Unknown"
            else:
                page = page + 1

            # Build context for the LLM
            context += (
                f"\n\nSOURCE: {filename}\n"
                f"PAGE: {page}\n"
                f"{document.page_content}\n"
            )

            # Deduplicate citations
            if filename not in seen:
                seen[filename] = {
                    "file": filename,
                    "pages": set(),
                    "score": score,
                }

            if page != "Unknown":
                seen[filename]["pages"].add(page)

            # Keep the best (lowest) distance score
            if score < seen[filename]["score"]:
                seen[filename]["score"] = score

        # Convert pages set to sorted list
        for item in seen.values():

            citations.append(
                {
                    "file": item["file"],
                    "pages": sorted(list(item["pages"])),
                }
            )

        return context, citations

    def answer(self, question):
        # First check relevance using similarity search.
        scored_docs = self.vector_db.similarity_search_with_score(
            question,
            k=3,
        )

        # If no documents are found, fail early with a clear response.
        if not scored_docs:
            return {
                "answer": "The provided source documents do not contain enough information to answer this question.",
                "citations": [],
            }

        # Lower score = better match (FAISS L2 distance).
        best_score = scored_docs[0][1]

        # Threshold (tune if needed).
        if best_score > 1.2:
            return {
                "answer": "The provided source documents do not contain enough information to answer this question.",
                "citations": [],
            }

        # Use MMR for better diversity and context coverage.
        docs = self.retrieve(question)

        if not docs:
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
            pages = ", ".join(str(p) for p in citation["pages"])
            print(
                f"""
        File : {citation['file']}
        Pages: {pages if pages else "N/A"}

        ----------------------------------------------------
        """
            )