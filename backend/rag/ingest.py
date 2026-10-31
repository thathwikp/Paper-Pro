# Takes a PDF and stores it in the vector database

import hashlib
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag.store import paper_collection, embed_texts

# Configuration
CHUNK_SIZE = 1000           # Num chars per chunk
CHUNK_OVERLAP = 150         # Shared chars
MIN_CHUNK_LENGTH = 100

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Create ID from the PDF's contents
def make_paper_id(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        file_bytes = pdf_file.read()
    full_hash = hashlib.sha256(file_bytes).hexdigest()
    return full_hash[:12]


# Check whether any chunks with this paper ID are already stored
def is_already_ingested(paper_id):
    existing = paper_collection.get(where={"paper_id": paper_id}, limit=1)
    return len(existing["ids"]) > 0


# Open PDF and split each page into chunks. Returns (page_number, chunk_text) pairs
def extract_page_chunks(pdf_path):
    page_chunks = []
    document = pymupdf.open(pdf_path)

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text()
        chunks_on_page = text_splitter.split_text(page_text)

        for chunk_text in chunks_on_page:
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                page_chunks.append((page_number, chunk_text))

    document.close()
    return page_chunks


# Ingest one PDF into the collection. Source is "library" or "upload"
def ingest_pdf(pdf_path, title, source="library", owner_id="public"):
    paper_id = make_paper_id(pdf_path)

    if is_already_ingested(paper_id):
        print(f"Skipping '{title}' (already in database)")
        return paper_id

    page_chunks = extract_page_chunks(pdf_path)

    # Scanned or image-only PDFs have no extractable text
    if not page_chunks:
        print(f"Skipping '{title}' (no extractable text)")
        return None

    # Build the three lists Chroma expects: IDs, texts, and metadata
    chunk_ids = []
    chunk_texts = []
    chunk_metadata = []

    for chunk_index, (page_number, chunk_text) in enumerate(page_chunks):
        chunk_ids.append(f"{paper_id}_{chunk_index}")
        chunk_texts.append(chunk_text)
        chunk_metadata.append({
            "paper_id": paper_id,
            "title": title,
            "page": page_number,
            "source": source,
            "owner_id": owner_id,
        })

    # Embed all chunks in one batch
    chunk_vectors = embed_texts(chunk_texts)
    paper_collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        metadatas=chunk_metadata,
        embeddings=chunk_vectors,
    )

    print(f"Ingested '{title}': {len(chunk_ids)} chunks")
    return paper_id