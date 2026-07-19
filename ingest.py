from ingestion.pdf_loader import load_documents

from ingestion.chunker import chunk_documents

from ingestion.pinecone_upload import upload
import pickle
from pro.keyword_based import KeywordSearchBM25
import os


docs = load_documents("docs2")

chunks = chunk_documents(docs)

upload(chunks)

#save bm25 model object
chunk_texts = [chunk.page_content for chunk in chunks]

# BM25 Build
keyword_search = KeywordSearchBM25(chunk_texts)

os.makedirs("data", exist_ok=True)

# BM25 Save
with open("data/bm25.pkl", "wb") as f:
    pickle.dump(keyword_search, f)

print("BM25 Saved Successfully")