# Builds the paper library: samples arXiv across many fields and years, then ingests the PDFs
# Usage (from project root): python -m backend.rag.build_library [--dry-run] [--limit N]
import argparse
import random
from collections import Counter

import arxiv

# Field -> arXiv categories. Edit freely to change what the library covers
CATEGORIES = {
    "Computer Science": ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.CR", "cs.DC", "cs.HC", "cs.RO"],
    "Mathematics": ["math.PR", "math.NT", "math.CO", "math.AP"],
    "Statistics": ["stat.ML", "stat.ME"],
    "Physics": ["physics.optics", "physics.bio-ph", "physics.ao-ph", "physics.soc-ph", "quant-ph", "hep-th"],
    "Astrophysics": ["astro-ph.GA", "astro-ph.CO"],
    "Condensed Matter": ["cond-mat.mes-hall", "cond-mat.mtrl-sci"],
    "Quantitative Biology": ["q-bio.NC", "q-bio.GN", "q-bio.PE"],
    "Economics": ["econ.GN", "econ.EM"],
    "Quantitative Finance": ["q-fin.ST", "q-fin.PM"],
    "Electrical Engineering": ["eess.SP", "eess.IV"],
}
YEARS = [2018, 2020, 2022, 2024, 2025]
PAPERS_PER_CELL = 2     # Papers per (category, year) pair
MAX_RANDOM_OFFSET = 200  # How far into a cell's results we may start, so reruns aren't all the newest papers


# List every (field, category, year) cell to sample from
def plan_cells(years=YEARS):
    return [
        (field, category, year)
        for field, categories in CATEGORIES.items()
        for category in categories
        for year in years
    ]


# arXiv query for papers whose primary category matches and that were submitted in the given year
def make_query(category, year):
    return f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"


def build_library(years=YEARS, per_cell=PAPERS_PER_CELL, limit=None, dry_run=False, seed=0):
    cells = plan_cells(years)
    if limit is not None:
        cells = random.Random(seed).sample(cells, min(limit, len(cells)))

    print(f"{len(cells)} cells x {per_cell} papers = up to {len(cells) * per_cell} papers")
    if dry_run:
        for field, category, year in cells:
            print(f"{field:25} {category:20} {year}")
        return

    # Imported late: these load the embedding model and open the database
    from backend.rag.ingest import ingest_pdf
    from backend.scripts.download_papers import download_papers

    rng = random.Random(seed)
    papers_per_field = Counter()
    failures = []

    for cell_number, (field, category, year) in enumerate(cells, start=1):
        print(f"\n[{cell_number}/{len(cells)}] {field} / {category} / {year}")
        try:
            papers = download_papers(
                make_query(category, year),
                max_results=per_cell,
                offset=rng.randrange(MAX_RANDOM_OFFSET),
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
        except Exception as error:  # a flaky API call shouldn't end an hour-long run
            print(f"Search failed: {error}")
            failures.append((category, year, str(error)))
            continue

        for paper in papers:
            try:
                paper_id = ingest_pdf(
                    paper["pdf_path"],
                    paper["title"],
                    source="library",
                    extra_metadata={
                        "arxiv_id": paper["arxiv_id"],
                        "field": field,
                        "category": paper["category"],
                        "year": paper["year"],
                    },
                )
            except Exception as error:
                print(f"Ingest failed for {paper['arxiv_id']}: {error}")
                failures.append((category, year, str(error)))
                continue
            if paper_id:
                papers_per_field[field] += 1

    # Imported here so --dry-run stays fast
    from backend.rag.store import paper_collection

    print("\n--- Summary ---")
    for field, count in sorted(papers_per_field.items()):
        print(f"{field:25} {count}")
    print(f"Papers: {sum(papers_per_field.values())}, failures: {len(failures)}")
    print(f"Chunks in database: {paper_collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    parser.add_argument("--per-cell", type=int, default=PAPERS_PER_CELL)
    parser.add_argument("--limit", type=int, help="only sample this many random cells (for trial runs)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without downloading")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_library(args.years, args.per_cell, args.limit, args.dry_run, args.seed)
