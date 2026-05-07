# 📄 Data Directory

Place your oncology PDF files inside `data/pdfs/`.

## Supported Files
- Medical textbooks (PDF)
- Clinical guidelines (PDF)
- Oncology research papers (PDF)

> ⚠️ PDF files are excluded from version control via `.gitignore` due to file size and copyright restrictions.

## How to Add Data
```bash
# Copy your PDFs into the folder
cp your_oncology_book.pdf data/pdfs/

# Then run ingestion
python ingest.py
```
