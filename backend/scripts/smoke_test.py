# End-to-end check: download a few papers, ingest them, then search using each paper's own abstract
# and confirm the top hit comes from that paper. Run from project root: python -m backend.scripts.smoke_test
from backend.rag.ingest import ingest_pdf
from backend.rag.store import paper_collection, embed_texts
from backend.scripts.download_papers import download_papers

QUERY = "retrieval augmented generation"
NUM_PAPERS = 4

papers = download_papers(QUERY, NUM_PAPERS)
paper_ids = {}
for paper in papers:
    paper_ids[paper["arxiv_id"]] = ingest_pdf(paper["pdf_path"], paper["title"])

print(f"\nCollection now holds {paper_collection.count()} chunks")

passed = 0
for paper in papers:
    # Use the first ~300 chars of the abstract as a natural-language query
    question = paper["abstract"][:300]
    result = paper_collection.query(query_embeddings=embed_texts([question]), n_results=3)
    top_ids = [metadata["paper_id"] for metadata in result["metadatas"][0]]
    expected = paper_ids[paper["arxiv_id"]]
    ok = top_ids[0] == expected
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {paper['title'][:70]}  (expected in top3: {expected in top_ids})")

print(f"\n{passed}/{len(papers)} papers retrieved as top hit")
