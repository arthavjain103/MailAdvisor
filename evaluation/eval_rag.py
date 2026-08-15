import json
import pickle
import sys


sys.stdout.reconfigure(encoding="utf-8")

from ragas import EvaluationDataset, evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq

from pro.semantic import SemanticSearch
from pro.hybrid import CombinedSearch
from llm.chat import generate_answer
import os

from ingestion.embedder import get_embedding_model

embedding_model = get_embedding_model()

# Wrap the Groq chat model for use as the Ragas evaluator LLM
evaluator_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
    )
)

# Wrap your existing HuggingFace embedding model for use in Ragas
wrapped_embeddings = LangchainEmbeddingsWrapper(embedding_model)


def initialize_pipeline():

    print("Loading BM25...")

    with open("data/bm25.pkl", "rb") as f:
        keyword_search = pickle.load(f)

    print("Connecting to Pinecone...")

    semantic_search = SemanticSearch()

    print("Initializing Hybrid Search...")

    hybrid_search = CombinedSearch(
        keyword_search=keyword_search,
        vector_search=semantic_search
    )

    return hybrid_search


def run_rag(hybrid_search, query):

    results = hybrid_search.search(
        query=query,
        top_n=3
    )

    # Final 5 reranked documents
    retrieved_contexts = [
        doc
        for score, doc in results
    ]

    # Context for LLM
    context = "\n\n---\n\n".join(
        retrieved_contexts
    )

    # Generate answer
    answer = generate_answer(
        query,
        context
    )

    return answer, retrieved_contexts


def main():

    # Initialize existing RAG pipeline
    hybrid_search = initialize_pipeline()

    # Load benchmark
    with open(
        "evaluation/test_questions.json",
        "r",
        encoding="utf-8"
    ) as f:
        test_questions = json.load(f)

    rows = []

    # Run every test question
    for item in test_questions:

        question = item["question"]
        reference = item["reference"]

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        answer, contexts = run_rag(
            hybrid_search,
            question
        )

        print("\nANSWER:")
        print(answer)

        print("\nRETRIEVED CONTEXTS:")

        for i, context in enumerate(contexts, 1):
            print(f"\n--- Context {i} ---")
            print(context)

        # Ragas row
        rows.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference
        })

    # Create Ragas dataset
    dataset = EvaluationDataset.from_list(rows)

    metrics = [
        Faithfulness(
            llm=evaluator_llm
        ),

        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=wrapped_embeddings
        ),

        ContextPrecision(
            llm=evaluator_llm
        ),

        ContextRecall(
            llm=evaluator_llm
        )
    ]

    # Evaluate
    print("\nRunning Ragas evaluation...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    print("\n" + "=" * 80)
    print("RAGAS RESULTS")
    print("=" * 80)

    print(result)


if __name__ == "__main__":
    main()