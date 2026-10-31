import os
import tempfile

# store.py opens the Chroma DB at import time, so point it at a throwaway
# directory before anything imports it
os.environ["PAPERPRO_CHROMA_PATH"] = tempfile.mkdtemp(prefix="paperpro_test_chroma_")
