"""
Ingestion: load PDFs, split them into chunks, embed them and store them in ChromaDB.
"""
from pathlib import Path
import hashlib
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Folders and database settings
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION = "university_knowledge"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_vectorstore():
    """Connect to the ChromaDB collection using the Qwen3 embedding model."""
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(DB_PATH),
    )


def ingest_pdf(uploaded_file):
    """Save and index a PDF uploaded through Streamlit."""
    pdf_bytes = uploaded_file.getvalue()

    # Check the file is really a PDF
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    # A unique fingerprint of the file, used to skip duplicates
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Make the filename safe to save
    filename = Path(uploaded_file.name).name
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    saved_name = f"{file_hash[:12]}_{filename}"
    pdf_path = DATA_DIR / saved_name

    vectorstore = get_vectorstore()

    # Skip the file if it has already been indexed
    if vectorstore.get(where={"file_hash": file_hash}, limit=1)["ids"]:
        return f"Already indexed: {filename}"

    pdf_path.write_bytes(pdf_bytes)

    try:
        # Load the PDF and split it into overlapping chunks
        documents = PyMuPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No extractable text found in PDF.")

        # Add metadata so each chunk can be traced back to its file
        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["filename"] = filename
            chunk.metadata["source"] = saved_name

        chunk_ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        vectorstore.add_documents(documents=chunks, ids=chunk_ids)

        return f"Successfully indexed {filename} ({len(chunks)} chunks)"

    except Exception:
        # Clean up anything partly saved if indexing fails
        partial = vectorstore.get(where={"file_hash": file_hash})
        if partial["ids"]:
            vectorstore.delete(ids=partial["ids"])
        pdf_path.unlink(missing_ok=True)
        raise


def list_documents():
    """Return the PDFs stored in the data folder."""
    return sorted(DATA_DIR.glob("*.pdf"))


def ingest_folder():
    """Index every PDF already in the data folder."""
    vectorstore = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for pdf_path in list_documents():
        file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        if vectorstore.get(where={"file_hash": file_hash}, limit=1)["ids"]:
            print("Already indexed:", pdf_path.name)
            continue

        chunks = splitter.split_documents(PyMuPDFLoader(str(pdf_path)).load())
        if not chunks:
            print("No text found (scanned PDF?):", pdf_path.name)
            continue

        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["filename"] = pdf_path.name
            chunk.metadata["source"] = pdf_path.name

        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        vectorstore.add_documents(documents=chunks, ids=ids)
        print(f"Indexed {pdf_path.name} ({len(chunks)} chunks)")


if __name__ == "__main__":
    ingest_folder()