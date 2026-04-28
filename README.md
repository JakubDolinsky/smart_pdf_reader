Smart PDF Reader (RAG-based AI system)

1. Overview
Smart PDF Reader is a Retrieval-Augmented Generation (RAG) application that allows users to ask for information from PDF documents using chat instead of reading them manually.
The system answers questions based on document content and supports multi-document analysis. It is designed as a local solution, running on a private server with locally deployed 
LLM and embedding models for improved data privacy. A key focus of the project is balancing accuracy, latency, and hardware limitations in a real-world CPU environment.

2. Key features
- chat-based interface for querying PDF documents
- multi-PDF ingestion and semantic search (RAG pipeline)
- local deployment (no external API dependency)
- source-grounded answers with document context
- conversation saved into DB (local SQL server)
- fallback mechanism for uncertain retrieval results
 
3. Architecture

The system is built as a multi-layer architecture:

user question - desktop client - REST API(.NET) - RAG API(Python, FastAPI) - RAG(Python, LLM, embedding)
									|						|
								SQL server 				 QDRANT	vector db

Data storage:
SQL Server → conversation history
Qdrant → vector embeddings (semantic search)

4. RAG pipeline

Ingestion:

- PDF parsing
- semantic chunking (chapter and paragraph-aware splitting)
- generation of embeddings
- storage in vector database (Qdrant)

Question answering:

- query embedding
- semantic retrieval (cosine similarity search)
- reranking of results
- prompt construction with context
- LLM generation (local model)

5. Model and design decisions
- two-stage retrieval system (cosine search + rerank) for getting the most relevant chunks
- hybrid chunking strategy (splitting into chunks by chapter, paragraphs and max token limitation)
- strict fallback logic for low-confidence retrievals (application returns "not mentioned in document" if the none of selected chunks reaches sufficient reranking score)
- CPU-optimized inference pipeline (GPU optional)
- balance challenge: lower latency vs model precision on consumer hardware (bigger LLM = more precise answer but higher latency)

6. Tech stack
- .NET (REST API + Desktop Client)
- Python (FastAPI, RAG pipeline)
- Qdrant (vector database)
- SQL Server
- Ollama (local LLM execution)
- Docker (full deployment orchestration)

7. Engineering challenges
- running LLM + embeddings efficiently on CPU-only hardware
- maintaining response quality under model size and hardware constraints
- designing robust chunking for long structured documents
- reducing hallucinations tuning retrieval pipeline + fallback logic
- balancing latency vs accuracy in real-time chat system (2–5 min inference on CPU)

8. Results and observations
- main bottleneck: LLM inference latency (CPU-bound)
- embedding + reranking quality strongly impacts final answer quality
- no clear reranking threshold was found for reliable chunk filtering, so the final threshold was chosen by balancing false positives and false negatives

9. Deployment
There is implemented automated local deployment via Docker + PowerShell scripts.

10. Summary
This project demonstrates an end-to-end implementation of a local RAG system, combining document processing, vector search, and LLM-based reasoning
in a production-style architecture.

Focus was placed on:

- real-world constraints (CPU-only environment)
- modular system design
- practical tradeoffs between performance and accuracy

11. Demo and repository
GitHub: https://github.com/JakubDolinsky/smart_pdf_reader
(Optional local deployment required – no hosted demo due to system complexity)

For more technical details check docs/technical.md file.