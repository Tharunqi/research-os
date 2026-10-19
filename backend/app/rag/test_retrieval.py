
from app.rag.retrieval import retrieve_chunks


def main():
    results = retrieve_chunks(
        question="What is the main idea behind gated linear units in Transformers?",
        paper_id=21,
        project_id=1,
        top_k=5,
    )

    print(f"\nRetrieved {len(results)} chunks\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Chunk {i} ---")
        print(f"Page: {result['page_number']}")
        print(f"Score: {result['score']}")
        print(result["text"][:700])
        print()


if __name__ == "__main__":
    main()
