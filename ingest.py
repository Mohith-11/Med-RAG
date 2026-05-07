import os
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from chunking.chunk import hierarchical_chunk
from embeddings.embed import embed_passages
from vectorstore.pineconeDB import upsert_chunks, reset_index


# 🔥 Folder containing all PDFs
PDF_FOLDER = "data/pdfs"


def load_all_pdfs(folder_path):
    documents = []

    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pdf")
    ]

    print(f"📚 Found {len(pdf_files)} PDF files")

    for pdf in pdf_files:
        print(f"\n📄 Loading: {os.path.basename(pdf)}")

        loader = PyPDFLoader(pdf)
        pages = loader.load()

        for page in pages:

            # 🔥 normalize whitespace (removes OCR artifacts, weird line breaks)
            text = " ".join(page.page_content.strip().split())

            # 🔥 remove useless pages / sections
            bad_sections = [
                "references",
                "bibliography",
                "acknowledgement",
                "acknowledgments",
                "table of contents",
                "index",
                "appendix",
                "copyright",
                "author index",
                "subject index",
                "list of figures",
                "list of tables",
                "about the author",
                "foreword",
                "preface"
            ]

            if any(x in text.lower() for x in bad_sections):
                continue

            # 🔥 skip tiny/noisy chunks
            if len(text.split()) < 40:
                continue

            documents.append({
                "text": text,
                "page": page.metadata.get("page", 0) + 1,   # 1-indexed
                "source": os.path.basename(pdf)
            })

    return documents


def process_batches(chunks, batch_size=100):

    total_batches = (len(chunks) + batch_size - 1) // batch_size

    print(f"\n📦 Total batches: {total_batches}")

    for i in tqdm(range(0, len(chunks), batch_size)):

        batch = chunks[i:i + batch_size]

        texts = [chunk["text"] for chunk in batch]

        # 🔥 create embeddings
        vectors = embed_passages(texts)

        # 🔥 store in Pinecone
        upsert_chunks(batch, vectors)

        print(f"✅ Processed batch {(i // batch_size) + 1}")


if __name__ == "__main__":

    print("🚀 Starting ingestion pipeline...\n")

    # 🔥 Reset Pinecone index (clean slate for fresh ingest)
    reset_index()

    # 🔥 Load PDFs
    documents = load_all_pdfs(PDF_FOLDER)

    print(f"\n✅ Loaded {len(documents)} cleaned pages")

    # 🔥 Hierarchical chunking
    chunks = hierarchical_chunk(documents)

    print(f"✅ Generated {len(chunks)} chunks")

    # 🔥 Store embeddings
    process_batches(chunks)

    print("\n🎉 Data successfully indexed in Pinecone")