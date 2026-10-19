
from app.rag.answer import answer_question


def main():
    result = answer_question(
        question=(
            "What are gated linear units, and what improvements "
            "does this paper report over standard activations?"
        ),
        paper_id=21,
        project_id=1,
        top_k=5,
    )

    print("\nANSWER\n")
    print(result["answer"])

    print("\nSOURCES\n")
    for source in result["sources"]:
        print(
            f"Paper ID: {source['paper_id']} | "
            f"Page: {source['page_number']} | "
            f"arXiv: {source['arxiv_id']}"
        )

    print(f"\nGrounded retrieval used: {result['grounded']}")


if __name__ == "__main__":
    main()
