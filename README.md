# ResearchMind AI

ResearchMind AI is a Retrieval-Augmented Generation (RAG) application that answers questions using a collection of research papers instead of relying only on a language model's internal knowledge.

The application indexes research PDFs into a FAISS vector database, retrieves the most relevant document sections for a user query, and generates an answer using Groq's Llama 3.3 model. Every answer includes the research papers used to generate the response along with the corresponding page numbers. If the uploaded documents do not contain enough information, the application explicitly reports that instead of generating unsupported answers.

---

## Project Overview

The objective of this project was to build an end-to-end research assistant capable of answering questions from a collection of research papers while maintaining traceability through citations.

The application follows a Retrieval-Augmented Generation workflow where document retrieval and language generation are separated. Rather than allowing the language model to answer from its pre-trained knowledge, the model is provided only with the context retrieved from the indexed research papers.

---

## Features

- Natural language question answering
- Semantic search over research papers
- FAISS vector database for document retrieval
- Maximum Marginal Relevance (MMR) retrieval
- Answer generation using Groq Llama 3.3
- Source citations with page numbers
- Detection of insufficient information
- PDF upload support
- Knowledge base rebuilding
- Streamlit-based web interface
- Chat history

---

## System Architecture

```
User Question
      │
      ▼
Sentence Transformer Embedding
      │
      ▼
FAISS Vector Search
      │
      ▼
MMR Retrieval
      │
      ▼
Retrieved Context
      │
      ▼
Groq Llama 3.3
      │
      ▼
Answer with Citations
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| User Interface | Streamlit |
| Framework | LangChain |
| Vector Database | FAISS |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Language Model | Groq Llama 3.3 70B |
| PDF Processing | PyPDFLoader |

---

## Project Structure

```
research-agent/

├── app.py
├── rag.py
├── build_vector_store.py
├── download_papers.py
├── config.py
├── prompts.py
├── utils.py
├── requirements.txt
├── README.md
├── sample_questions.md
├── tradeoffs.md
├── .env.example

├── documents/
├── vector_store/
├── screenshots/
├── outputs/
└── tests/
```

---

## Prerequisites

- Python 3.11 or later
- Git
- Groq API Key

---

## Installation

Clone the repository.

```bash
git clone https://github.com/Svka9743/research-agent.git

cd research-agent
```

Create and activate a virtual environment.

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a file named `.env`.

```
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Application

Download sample papers (optional).

```bash
python download_papers.py
```

Build the FAISS index.

```bash
python build_vector_store.py
```

Start the Streamlit application.

```bash
streamlit run app.py
```

---

## Sample Questions

Examples:

- What is Retrieval-Augmented Generation?
- Explain Transformers.
- What are AI Agents?
- Explain Prompt Engineering.
- Explain Federated Learning.
- Explain Diffusion Models.
- Explain Large Language Models.

Negative test:

```
Who is Virat Kohli?
```

Expected output:

```
The provided source documents do not contain enough information to answer this question.
```

---

## Retrieval Pipeline

The application follows the steps below:

1. Convert the user question into an embedding.
2. Retrieve relevant document chunks from the FAISS vector database.
3. Apply Maximum Marginal Relevance to reduce redundant context.
4. Combine the retrieved chunks into a context window.
5. Generate the final answer using Groq Llama 3.3.
6. Display the answer together with source citations.
7. Return an insufficient-information response when no relevant context is found.

---

## Screenshots

The following screenshots are included in the `screenshots` directory.

- Home page
- Chat interface
- Research answer
- Citation display
- Upload documents
- Sidebar
- No-answer example

---

## Limitations

- The application works only with the uploaded PDF documents.
- Web search is not included.
- Scanned PDFs without extractable text are not processed.
- The vector index must be rebuilt after adding new documents.
- Retrieval quality depends on the quality of the uploaded documents.

---

## Future Improvements

Possible extensions include:

- Incremental indexing
- Hybrid retrieval combining keyword and semantic search
- Cross-encoder re-ranking
- OCR support for scanned PDFs
- Streaming responses
- Multi-document summarization
- Automatic indexing after uploads
- Cloud deployment

---

## Assessment Deliverables

This submission includes:

- Research Agent with Citations
- Source document retrieval
- Citation-based answers
- Streamlit interface
- Sample questions
- Retrieval workflow documentation
- Tradeoff documentation
- Detection of insufficient information

---

## Author

VINAY KUMAR S

AI Engineering Assessment Project

---

## License

This project was developed as part of an AI Engineering assessment and is intended for educational and demonstration purposes.

## Demo 

```bash

https://research-agent-hgftpp2haaj9vammgzcczu.streamlit.app/

```
