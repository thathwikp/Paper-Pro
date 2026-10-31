import numpy as np
import pymupdf
import pytest

from backend.rag import ingest
from backend.rag.store import paper_collection, embed_texts, CHROMA_PATH


def make_pdf(path, pages):
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=10)
    document.save(path)
    document.close()
    return str(path)


def long_text(topic, sentences=40):
    return " ".join(f"This sentence number {i} is about {topic}." for i in range(sentences))


@pytest.fixture(autouse=True)
def clean_collection():
    existing = paper_collection.get()["ids"]
    if existing:
        paper_collection.delete(ids=existing)
    yield


def test_uses_temp_database():
    assert "paperpro_test_chroma_" in CHROMA_PATH


def test_make_paper_id_is_deterministic_and_content_based(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", [long_text("cats")])
    a_copy = make_pdf(tmp_path / "a_copy.pdf", [long_text("cats")])
    b = make_pdf(tmp_path / "b.pdf", [long_text("dogs")])

    assert ingest.make_paper_id(a) == ingest.make_paper_id(a)
    assert len(ingest.make_paper_id(a)) == 12
    assert ingest.make_paper_id(a) != ingest.make_paper_id(b)
    assert ingest.make_paper_id(a_copy) != ingest.make_paper_id(b)


def test_extract_page_chunks_sizes_and_page_numbers(tmp_path):
    pdf = make_pdf(tmp_path / "two.pdf", [long_text("cats"), long_text("dogs")])
    chunks = ingest.extract_page_chunks(pdf)

    assert chunks
    assert {page for page, _ in chunks} == {1, 2}  # 1-indexed
    for _, text in chunks:
        assert ingest.MIN_CHUNK_LENGTH <= len(text) <= ingest.CHUNK_SIZE
    assert all("cats" in t for p, t in chunks if p == 1)
    assert all("dogs" in t for p, t in chunks if p == 2)


def test_short_text_yields_no_chunks(tmp_path):
    pdf = make_pdf(tmp_path / "short.pdf", ["Too short."])
    assert ingest.extract_page_chunks(pdf) == []


def test_ingest_stores_chunks_with_metadata(tmp_path):
    pdf = make_pdf(tmp_path / "p.pdf", [long_text("cats"), long_text("dogs")])
    expected = len(ingest.extract_page_chunks(pdf))

    paper_id = ingest.ingest_pdf(pdf, "Cats and Dogs", source="upload", owner_id="u1")

    assert paper_collection.count() == expected
    stored = paper_collection.get(where={"paper_id": paper_id})
    for metadata in stored["metadatas"]:
        assert metadata["title"] == "Cats and Dogs"
        assert metadata["source"] == "upload"
        assert metadata["owner_id"] == "u1"
        assert metadata["page"] in (1, 2)


def test_ingest_twice_is_skipped(tmp_path):
    pdf = make_pdf(tmp_path / "p.pdf", [long_text("cats")])
    ingest.ingest_pdf(pdf, "Cats")
    count = paper_collection.count()

    ingest.ingest_pdf(pdf, "Cats again, different title")

    assert paper_collection.count() == count


def test_pdf_without_text_does_not_crash(tmp_path):
    pdf = make_pdf(tmp_path / "blank.pdf", [""])
    assert ingest.ingest_pdf(pdf, "Blank") is None
    assert paper_collection.count() == 0


def test_embeddings_shape_and_similarity():
    vectors = embed_texts([
        "A cat sat on the mat.",
        "A kitten is resting on the rug.",
        "Quarterly revenue grew by twelve percent.",
    ])
    assert len(vectors) == 3 and len(vectors[0]) == 384
    similar = np.dot(vectors[0], vectors[1])
    unrelated = np.dot(vectors[0], vectors[2])
    assert similar > unrelated


def test_query_finds_the_right_paper(tmp_path):
    cats = ingest.ingest_pdf(make_pdf(tmp_path / "c.pdf", [long_text("feline cats and whiskers")]), "Cats")
    finance = ingest.ingest_pdf(make_pdf(tmp_path / "f.pdf", [long_text("stock market investing")]), "Finance")

    result = paper_collection.query(query_embeddings=embed_texts(["stock market investing"]), n_results=1)

    assert result["metadatas"][0][0]["paper_id"] == finance
    assert result["metadatas"][0][0]["paper_id"] != cats
