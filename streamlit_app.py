import streamlit as st
import pickle
import os
from pathlib import Path
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.insert(0, str(Path(__file__).parent))

from pro.keyword_based import KeywordSearchBM25
from pro.semantic import SemanticSearch
from pro.hybrid import CombinedSearch
from llm.chat import generate_answer

st.set_page_config(page_title="Email RAG Search", layout="wide")


@st.cache_resource
def load_bm25_model():
    if not os.path.exists("data/bm25.pkl"):
        return None
    with open("data/bm25.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_semantic_search():
    return SemanticSearch()


@st.cache_resource
def load_hybrid_search(_keyword_search, _vector_search):
    return CombinedSearch(keyword_search=_keyword_search, vector_search=_vector_search)


if "hybrid_search" not in st.session_state:
    st.session_state.hybrid_search = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# Sidebar - config + retrieved chunks
st.sidebar.title("Configuration")

if st.sidebar.button("Initialize Models"):
    with st.spinner("Loading models..."):
        keyword_search = load_bm25_model()
        if keyword_search is None:
            st.sidebar.error("BM25 model not found at data/bm25.pkl")
        else:
            semantic_search = load_semantic_search()
            st.session_state.hybrid_search = load_hybrid_search(keyword_search, semantic_search)
            st.session_state.models_loaded = True
            st.sidebar.success("Models loaded")

top_n = st.sidebar.slider("Number of Results", min_value=1, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieved Chunks")

if st.session_state.search_results:
    for rank, (score, doc) in enumerate(st.session_state.search_results, start=1):
        with st.sidebar.expander(f"#{rank} — score: {score:.4f}"):
            st.text(doc)
else:
    st.sidebar.caption("No chunks yet. Search to see them here.")

# Main content
st.title("MailAdvisor : Your Email AI Assistant")

if not st.session_state.models_loaded:
    st.warning("Please initialize models first using the sidebar button")
else:
    query = st.text_area("Enter your query:", height=100)

    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("Search", type="primary")
    with col2:
        clear_button = st.button("Clear")

    if clear_button:
        st.session_state.search_results = None
        st.session_state.answer = None
        st.rerun()

    if search_button and query:
        with st.spinner("Searching..."):
            results = st.session_state.hybrid_search.search(query=query, top_n=top_n)
            st.session_state.search_results = results
            context = "".join([doc for score, doc in results])

        with st.spinner("Generating answer..."):
            st.session_state.answer = generate_answer(query, context)
    elif search_button and not query:
        st.warning("Please enter a query")

    if st.session_state.answer:
        st.subheader("Answer")
        st.markdown(st.session_state.answer)