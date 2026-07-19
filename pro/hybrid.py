import numpy as np
from reranker import CrossEncoderReranker
class CombinedSearch:
    def __init__(self, keyword_search, vector_search,):
       
        self.keyword_search = keyword_search
        self.vector_search = vector_search
        self.k = 60
        self.reranker = CrossEncoderReranker()


        
    def search(self, query, top_n=5):
        """
        Perform a combined search using both keyword-based and vector-based search,
        and return re-ranked results.

        Parameters:
        - query (str): The search query for the keyword-based search.
        - top_n (int): The number of top results to return.

        Returns:
        - list of tuple: Ranked results with combined similarity scores and document text.
        """
        fetch_k = top_n * 5
       # BM25 Results
        keyword_results = self.keyword_search.search(query, fetch_k)

        # Pinecone Results
        semantic_results = self.vector_search.search(query, fetch_k)

        # Stores final RRF score
        scores = {}

        # BM25 Ranking
      
        for rank, (score, doc) in enumerate(keyword_results, start=1):

            text = doc if isinstance(doc, str) else doc.page_content

            rrf_score = 1 / (self.k + rank)

            scores[text] = scores.get(text, 0) + rrf_score

        # Semantic Ranking
      
        for rank, (doc, score) in enumerate(semantic_results, start=1):

            text = doc if isinstance(doc, str) else doc.page_content

            rrf_score = 1 / (self.k + rank)

            scores[text] = scores.get(text, 0) + rrf_score
            
        

      
        # Sort according to RRF score
        ranked_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take Top 25 after RRF
        candidate_docs = [doc for doc, score in ranked_results[:25]]

        # Cross Encoder Re-ranking
        reranked = self.reranker.rerank(
            query,
            candidate_docs,
            top_n=top_n
        )

        return reranked