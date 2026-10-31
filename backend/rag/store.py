# Sets up embedding model and Chroma vector database
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Configuration. Path is anchored to the project root so it doesn't depend on the working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = os.environ.get("PAPERPRO_CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma_db"))
COLLECTION_NAME = "papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load model and create Chroma client
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
paper_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Turn list of strings into a list of vectors
def embed_texts(texts):
    vectors = embedding_model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
