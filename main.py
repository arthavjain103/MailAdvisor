from ingestion.pdf_loader import load_documents
from ingestion.chunker import chunk_documents

from pro.keyword_based import KeywordSearchBM25
from pro.semantic import SemanticSearch
from pro.hybrid import CombinedSearch
from llm.chat import generate_answer
import pickle


def main():

    print("Loading PDFs...")

    print("Initializing BM25...")
    with open("data/bm25.pkl", "rb") as f:
       keyword_search = pickle.load(f)
   
    print("loaded bm25 model successfully  ")
    print("Connecting to Pinecone...")

    semantic_search = SemanticSearch()

    print("Initializing Hybrid Search...")

    hybrid_search = CombinedSearch(
        keyword_search=keyword_search,
        vector_search=semantic_search,

    )

    while True:

        query = input("Ask Question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        # Retrieve relevant chunks
        results = hybrid_search.search(query=query, top_n=5)

        # Build context from retrieved chunks
        context = "".join([doc for score, doc in results])

        print("Generating answer with Groq...")

        # Generate final answer
        answer = generate_answer(query, context)

       
        print(answer)

        print("" + "=" * 80)
        print("RETRIEVED CHUNKS")
        print("=" * 80)

        for rank, (score, doc) in enumerate(results, start=1):
            print(f"Rank {rank} | Score: {score:.4f}")
            print(doc[:300])
            print("-" * 80)


if __name__ == "__main__":
    main()