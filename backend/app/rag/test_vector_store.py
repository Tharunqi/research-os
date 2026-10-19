
from app.rag.vector_store import (
    get_database_config,
    get_vector_store,
    configure_rag,
)


def main():
    print("1. Checking database configuration...")
    config = get_database_config()
    print(f"   Database: {config['database']}")
    print(f"   Host: {config['host']}:{config['port']}")

    print("\n2. Initializing PostgreSQL vector store...")
    vector_store = get_vector_store()
    print(f"   Vector store initialized: {type(vector_store).__name__}")
    print("   Table configured: researchos_chunks")

    print("\n3. Configuring embedding model...")
    configure_rag()
    print("   Embedding model configured successfully.")

    print("\nAll initialization checks passed!")


if __name__ == "__main__":
    main()
