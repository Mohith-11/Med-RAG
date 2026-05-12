# ============================================================
# Med-RAG - 45 Commits Push Script
# Run from: c:\Users\mohit\Documents\QA\rag_project
# ============================================================

Set-Location "c:\Users\mohit\Documents\QA\rag_project"

# Git init & remote
git init
git remote remove origin 2>$null
git remote add origin "https://github.com/Mohith-11/Med-RAG.git"
git branch -M main

# ---- SETUP (1-9) ----
git commit --allow-empty -m "chore: initialize project repository"

git add ".gitignore"
git commit -m "chore: add .gitignore for Python and data files"

git add ".env.example"
git commit -m "chore: add environment variable template"

git add "requirements.txt"
git commit -m "chore: add project dependencies"

git add "config.py"
git commit -m "chore: add centralized project configuration"

git add "README.md"
git commit -m "docs: add project README with architecture and setup"

git add "CONTRIBUTING.md"
git commit -m "docs: add contributing guidelines"

git add "LICENSE"
git commit -m "docs: add MIT license"

git add ".github\PULL_REQUEST_TEMPLATE.md"
git commit -m "chore: add GitHub pull request template"

# ---- DATA LAYER (10-11) ----
git add "data\pdfs\.gitkeep"
git commit -m "chore: add data directory structure placeholder"

git add "data\README.md"
git commit -m "docs: add data directory usage instructions"

# ---- LOADERS (12-13) ----
git add "loaders\pdf_loader.py"
git commit -m "feat: add PDF loader module using PyPDF"

git add "loaders\__init__.py"
git commit -m "feat: initialize loaders package"

# ---- PREPROCESSING (14-15) ----
git add "preprocessing\clean.py"
git commit -m "feat: add text cleaning and normalization module"

git add "preprocessing\__init__.py"
git commit -m "feat: initialize preprocessing package"

# ---- CHUNKING (16-17) ----
git add "chunking\chunk.py"
git commit -m "feat: add hierarchical chunking with metadata"

git add "chunking\__init__.py"
git commit -m "feat: initialize chunking package"

# ---- EMBEDDINGS (18-19) ----
git add "embeddings\__init__.py"
git commit -m "feat: initialize embeddings package"

git add "embeddings\embed.py"
git commit -m "feat: add sentence-transformer passage embedding"

# ---- VECTOR STORE (20-24) ----
git add "vectorstore\__init__.py"
git commit -m "feat: initialize vectorstore package"

git add "vectorstore\pineconeDB.py"
git commit -m "feat: add Pinecone index management and chunk upsert"

git add "vectorstore\query.py"
git commit -m "feat: add vector similarity search query module"

git add "vectorstore\query_rewrite\__init__.py"
git commit -m "feat: initialize query_rewrite subpackage"

git add "vectorstore\query_rewrite\rewrite.py"
git commit -m "feat: add LLM-based query rewriting"

# ---- INGESTION PIPELINE (25) ----
git add "ingest.py"
git commit -m "feat: add full PDF ingestion and indexing pipeline"

# ---- RETRIEVAL (26-34) ----
git add "retrieval\__init__.py"
git commit -m "feat: initialize retrieval package"

git add "retrieval\retrieve.py"
git commit -m "feat: add base vector retrieval module"

git add "retrieval\crag.py"
git commit -m "feat: add CRAG corrective multi-query retrieval"

git add "retrieval\reasoning.py"
git commit -m "feat: add sub-query decomposition and reasoning"

git add "retrieval\rerank.py"
git commit -m "feat: add cross-encoder reranking module"

git add "retrieval\filter.py"
git commit -m "feat: add metadata-based result filtering"

git add "retrieval\compress.py"
git commit -m "feat: add context compression with citation tracking"

git add "retrieval\context.py"
git commit -m "feat: add context assembly utilities"

# ---- GENERATOR (35-39) ----
git add "generator\__init__.py"
git commit -m "feat: initialize generator package"

git add "generator\generate.py"
git commit -m "feat: add LLM-powered answer generation"

git add "generator\med_gemma.py"
git commit -m "feat: add Medical Gemma model integration"

git add "generator\verify.py"
git commit -m "feat: add answer verification pipeline"

# ---- UTILS (40) ----
git add "utils\__init__.py" "utils\helpers.py"
git commit -m "feat: add utility helper functions (truncate, format, deduplicate)"

# ---- EVALUATION (41-43) ----
git add "evaluation\__init__.py"
git commit -m "feat: initialize evaluation package"

git add "evaluation\full_eval.py"
git commit -m "feat: add full evaluation suite (RAGAS, BERTScore, ROUGE)"

git add "evaluate_rag.py"
git commit -m "feat: add RAG evaluation runner script"

# ---- GROUND TRUTH (44) ----
git add "generate_ground_truths.py"
git commit -m "feat: add oncology Q&A ground truth generation"

# ---- MAIN APP (45) ----
git add "app.py"
git commit -m "feat: add main application entry point with full pipeline"

# ---- PUSH ----
Write-Host "`n✅ All 45 commits created. Pushing to GitHub..." -ForegroundColor Green
git push -u origin main --force

Write-Host "`n🎉 Successfully pushed to https://github.com/Mohith-11/Med-RAG" -ForegroundColor Cyan
