# 🧠 Oncology RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline specialized for oncology knowledge. Built with Pinecone, LangChain, and Sentence Transformers, this system answers medical questions by retrieving relevant content from a curated set of oncology textbooks and guidelines.

---

## 📌 Features

- 📄 **Multi-PDF ingestion** with smart cleaning (removes noise, boilerplate, short pages)
- ✂️ **Hierarchical chunking** for better semantic coverage
- 🔍 **CRAG (Corrective RAG)** retrieval with multi-query support
- 🔀 **Query rewriting** for improved retrieval precision
- 🧩 **Sub-query decomposition** via multi-step reasoning
- 📊 **Metadata filtering** and **cross-encoder reranking**
- 🗜️ **Context compression** with citation tracking
- ✅ **Answer verification** pipeline
- 📈 **Comprehensive evaluation** (RAGAS, BERTScore, ROUGE, LLM-as-a-Judge)

---

## 🗂️ Project Structure

```
rag_project/
│
├── app.py                        # 🚀 Main entry point (ask a question)
├── ingest.py                     # 📥 PDF ingestion & indexing pipeline
├── evaluate_rag.py               # 📊 RAG evaluation runner
├── generate_ground_truths.py     # 🏷️ Ground truth Q&A generation
├── requirements.txt              # 📦 Python dependencies
├── .env                          # 🔐 API keys (NOT pushed to GitHub)
│
├── chunking/
│   └── chunk.py                  # Hierarchical chunking logic
│
├── embeddings/
│   └── embed.py                  # Passage embedding (Sentence Transformers)
│
├── loaders/
│   └── pdf_loader.py             # PDF loading utilities
│
├── preprocessing/
│   └── clean.py                  # Text cleaning & normalization
│
├── vectorstore/
│   ├── pineconeDB.py             # Pinecone upsert & index management
│   ├── query.py                  # Vector similarity search
│   └── query_rewrite/
│       └── rewrite.py            # Query rewriting using LLM
│
├── retrieval/
│   ├── retrieve.py               # Base retrieval logic
│   ├── crag.py                   # Corrective RAG (multi-query)
│   ├── rerank.py                 # Cross-encoder reranking
│   ├── compress.py               # Context compression
│   ├── filter.py                 # Metadata-based filtering
│   ├── context.py                # Context assembly
│   └── reasoning.py              # Sub-query decomposition
│
├── generator/
│   ├── generate.py               # Answer generation (LLM)
│   ├── med_gemma.py              # Medical Gemma model integration
│   └── verify.py                 # Answer verification
│
├── evaluation/
│   └── full_eval.py              # Full evaluation suite
│
└── data/
    └── pdfs/                     # 📄 Oncology PDFs (NOT pushed to GitHub)
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag_project.git
cd rag_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (see `.env.example`):

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
OPENAI_API_KEY=your_openai_api_key     # or whichever LLM you use
```

### 5. Add your PDF documents

Place your oncology PDFs inside `data/pdfs/`.

---

## 🚀 Usage

### Ingest PDFs into Pinecone

```bash
python ingest.py
```

This will:
1. Load and clean all PDFs from `data/pdfs/`
2. Apply hierarchical chunking
3. Generate embeddings
4. Upsert vectors into Pinecone

### Ask a Medical Question

```bash
python app.py
```

Enter your question when prompted. The system will:
1. Rewrite & decompose the query
2. Retrieve relevant chunks via CRAG
3. Filter, rerank, and compress context
4. Generate and verify the answer with citations

### Run Evaluation

```bash
python evaluate_rag.py
```

---

## 📊 Evaluation Metrics

| Metric | Framework |
|--------|-----------|
| Faithfulness | RAGAS |
| Answer Relevancy | RAGAS |
| Context Precision | RAGAS |
| Semantic Similarity | BERTScore |
| N-gram Overlap | ROUGE |
| LLM-as-a-Judge | Custom prompt |

---



---

## 📦 Dependencies

- `pinecone >= 5.0.1`
- `sentence-transformers >= 2.2.2`
- `langchain >= 0.1.0`
- `pypdf >= 4.0.0`
- `python-dotenv >= 1.0.0`
- `tqdm >= 4.66.0`
- `numpy >= 1.24.0`

---

## 📝 License

This project is for research and educational purposes only. Medical PDFs used for ingestion remain under their respective copyrights and are **not included** in this repository.
