import os
import pickle
from ingestion.pdf_loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.pinecone_upload import upload
from pro.keyword_based import KeywordSearchBM25

# 1. Load PDFs with merged contiguous text & initial metadata
docs = load_documents("docs2")

# 2. Hybrid Routing Chunking
chunks = chunk_documents(docs)

# 3. Vector DB Upload
upload(chunks)

# 4. Extract Text & Rich Metadata for BM25
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_metadatas = [chunk.metadata for chunk in chunks]

# 5. BM25 Build & Save
keyword_search = KeywordSearchBM25(chunk_texts, metadatas=chunk_metadatas)

os.makedirs("data", exist_ok=True)
with open("data/bm25.pkl", "wb") as f:
    pickle.dump(keyword_search, f)

print("✅ Ingestion complete & BM25 Model Saved Successfully!")