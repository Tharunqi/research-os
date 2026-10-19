
from app.services.arxiv import search_papers


def main():
    # Test 1: Exact arXiv ID
    results = search_papers("1706.03762", limit=5)

    assert len(results) == 1, f"Expected one paper, got {len(results)}"

    paper = results[0]
    print("Exact ID result:")
    print("Title:", paper["title"])
    print("arXiv ID:", paper["arxiv_id"])

    assert paper["title"] == "Attention Is All You Need"

    # Test 2: Exact title
    results = search_papers("Attention Is All You Need", limit=5)

    assert results, "No paper returned for exact title"

    exact_matches = [
        paper for paper in results
        if paper["title"] == "Attention Is All You Need"
    ]

    assert exact_matches, "Exact title was not returned"

    print("\nExact title result:")
    print("Title:", exact_matches[0]["title"])
    print("arXiv ID:", exact_matches[0]["arxiv_id"])

    print("\nAll discovery tests passed!")


if __name__ == "__main__":
    main()
