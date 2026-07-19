from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, documents, top_n=5):

        pairs = [(query, doc) for doc in documents]

        scores = self.model.predict(pairs)

        results = list(zip(scores, documents))

        results.sort(reverse=True)

        return results[:top_n]