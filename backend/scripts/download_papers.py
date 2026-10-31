# Downloads open-access papers from arXiv into data/pdfs, and optionally ingests them
# Usage (from project root): python -m backend.scripts.download_papers --query "retrieval augmented generation" --n 5 --ingest
import argparse
import json
from pathlib import Path

import arxiv
import requests

PDF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"
MANIFEST_PATH = PDF_DIR / "manifest.json"


# Download the top results for a query. Returns a list of dicts describing each paper
def download_papers(query, max_results=5):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(delay_seconds=3)  # arXiv asks for at most 1 request per 3 seconds
    search = arxiv.Search(query=query, max_results=max_results)

    papers = []
    for result in client.results(search):
        filename = f"{result.get_short_id()}.pdf"
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            response = requests.get(result.pdf_url, timeout=60)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
        papers.append({
            "arxiv_id": result.get_short_id(),
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "abstract": result.summary,
            "pdf_path": str(pdf_path),
        })
        print(f"Downloaded {result.get_short_id()}: {result.title}")

    # Merge with any earlier manifest
    known = {}
    if MANIFEST_PATH.exists():
        known = {p["arxiv_id"]: p for p in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}
    known.update({p["arxiv_id"]: p for p in papers})
    MANIFEST_PATH.write_text(json.dumps(list(known.values()), indent=2), encoding="utf-8")
    return papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--ingest", action="store_true", help="also add the PDFs to the vector database")
    args = parser.parse_args()

    downloaded = download_papers(args.query, args.n)
    if args.ingest:
        from backend.rag.ingest import ingest_pdf  # imported late: loads the embedding model
        for paper in downloaded:
            ingest_pdf(paper["pdf_path"], paper["title"], source="library")
