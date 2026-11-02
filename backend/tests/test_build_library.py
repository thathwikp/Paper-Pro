import pytest

from backend.rag import build_library
from backend.scripts import download_papers


def test_plan_cells_covers_every_field_category_and_year():
    cells = build_library.plan_cells()
    expected = sum(len(c) for c in build_library.CATEGORIES.values()) * len(build_library.YEARS)

    assert len(cells) == expected
    assert len(set(cells)) == len(cells)
    assert {field for field, _, _ in cells} == set(build_library.CATEGORIES)
    assert {year for _, _, year in cells} == set(build_library.YEARS)


def test_make_query_filters_category_and_year():
    query = build_library.make_query("cs.CL", 2022)
    assert query == "cat:cs.CL AND submittedDate:[202201010000 TO 202212312359]"


def test_strip_version():
    assert download_papers.strip_version("2506.06962v3") == "2506.06962"
    assert download_papers.strip_version("2506.06962") == "2506.06962"


def test_fetch_pdf_rejects_non_pdf(monkeypatch):
    class FakeResponse:
        content = b"<html>Rate exceeded</html>"
        def raise_for_status(self):
            pass

    monkeypatch.setattr(download_papers.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError):
        download_papers.fetch_pdf("https://arxiv.org/pdf/x")


def test_extra_metadata_is_stored(tmp_path):
    import pymupdf
    from backend.rag import ingest
    from backend.rag.store import paper_collection

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), "Galaxies form in dark matter halos. " * 30, fontsize=10)
    pdf = tmp_path / "g.pdf"
    document.save(pdf)

    paper_id = ingest.ingest_pdf(str(pdf), "Galaxies", extra_metadata={"field": "Astrophysics", "year": 2024})

    metadata = paper_collection.get(where={"paper_id": paper_id}, limit=1)["metadatas"][0]
    assert metadata["field"] == "Astrophysics" and metadata["year"] == 2024
    paper_collection.delete(where={"paper_id": paper_id})
