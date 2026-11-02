# Downloads open-access papers from arXiv into data/pdfs, and optionally ingests them
# Usage (from project root): python -m backend.scripts.download_papers --query "retrieval augmented generation" --n 5 --ingest
import argparse
import json
import re
import time
from pathlib import Path

import arxiv
import requests

PDF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"
MANIFEST_PATH = PDF_DIR / "manifest.json"
DOWNLOAD_DELAY_SECONDS = 3  # arXiv asks for at most 1 request per 3 seconds


# "2506.06962v3" -> "2506.06962", so different versions of a paper share one file
def strip_version(short_id):
    return re.sub(r"v\d+$", "", short_id)


# Fetch a PDF, checking it really is one (rate-limited responses come back as HTML)
def fetch_pdf(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Response from {url} is not a PDF")
    return response.content


# Download the results for a query. Returns a list of dicts describing each paper
def download_papers(query, max_results=5, offset=0, sort_by=arxiv.SortCriterion.Relevance):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(delay_seconds=DOWNLOAD_DELAY_SECONDS)
    # In arxiv 4.x max_results counts from the start of the results, not from the offset
    search = arxiv.Search(query=query, max_results=offset + max_results, sort_by=sort_by)

    papers = []
    for result in client.results(search, offset=offset):
        arxiv_id = strip_version(result.get_short_id())
        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

        # Only download if PDF doesn't already exist
        already_downloaded = pdf_path.exists()
        if not already_downloaded:
            try:
                pdf_path.write_bytes(fetch_pdf(result.pdf_url))
            except (requests.RequestException, ValueError) as error:
                print(f"Failed to download {arxiv_id}: {error}")
                continue
            time.sleep(DOWNLOAD_DELAY_SECONDS)

        papers.append({
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "category": result.primary_category,
            "abstract": result.summary,
            "pdf_path": str(pdf_path),
        })

        if already_downloaded:
            print(f"Fetched {arxiv_id} from cache: {result.title}")
        else:
            print(f"Downloaded {arxiv_id}: {result.title}")

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
