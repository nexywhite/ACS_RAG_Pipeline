import os
from preprocess_loaders import load_pdf, load_text, ocr_image, split_documents

from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


# DIRECTORIES TO INDEX
PDF_DIR = "data/pdfs"
TEXT_DIR = "data/text"
IMAGE_DIR = "data/images"

# OUTPUT VECTOR DB DIRECTORY
CHROMA_DIR = "chroma_db"


def load_all_documents():
    all_docs = []

    # Load PDFs
    if os.path.isdir(PDF_DIR):
        for filename in os.listdir(PDF_DIR):
            if filename.endswith(".pdf"):
                path = os.path.join(PDF_DIR, filename)
                print(f"Loading PDF: {path}")
                all_docs.extend(load_pdf(path))

    # Load TEXT
    if os.path.isdir(TEXT_DIR):
        for filename in os.listdir(TEXT_DIR):
            if filename.endswith(".txt") or filename.endswith(".md"):
                path = os.path.join(TEXT_DIR, filename)
                print(f"Loading text: {path}")
                all_docs.extend(load_text(path))

    # Load IMAGES
    if os.path.isdir(IMAGE_DIR):
        for filename in os.listdir(IMAGE_DIR):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(IMAGE_DIR, filename)
                print(f"Loading image: {path}")
                all_docs.extend(ocr_image(path))

    return all_docs


def build_chroma_index():
    print("Step 1: Loading documents...")
    docs = load_all_documents()

    print(f"Loaded {len(docs)} raw documents.")

    print("Step 2: Splitting into chunks...")
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=100)
    print(f"Generated {len(chunks)} chunks.")

    print("Step 3: Creating embeddings model...")
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Step 4: Building Chroma vector database...")
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=CHROMA_DIR
    )

    print("Saving vector DB...")
    vectorstore.persist()

    print("Indexing complete!")
    print(f"Database saved in: {CHROMA_DIR}")


if __name__ == "__main__":
    build_chroma_index()
