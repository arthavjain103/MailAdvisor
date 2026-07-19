# Email RAG System

An intelligent Retrieval-Augmented Generation (RAG) system designed to answer questions about email content and documents using hybrid search and large language models.

## Project Overview

The Email RAG system processes PDF documents and allows users to query them through an intelligent Q&A interface. It combines multiple search strategies (keyword-based and semantic search) with advanced re-ranking and LLM-powered answer generation to provide accurate, contextual responses.

## Architecture

### System Components

```
email_rag/
├── ingestion/              # Document processing pipeline
│   ├── pdf_loader.py      # Load PDFs from directory
│   ├── chunker.py         # Split documents into chunks
│   ├── embedder.py        # Generate embeddings using HuggingFace
│   └── pinecone_upload.py # Upload embeddings to Pinecone
├── pro/                    # Search strategies
│   ├── keyword_based.py   # BM25 keyword search implementation
│   ├── semantic.py        # Vector similarity search via Pinecone
│   └── hybrid.py          # Combined search with re-ranking
├── llm/
│   └── chat.py            # LLM-powered answer generation
├── config.py              # Configuration parameters
├── ingest.py              # Document ingestion script
└── main.py                # Main interactive Q&A interface
```

### Data Flow

1. **Ingestion Phase** (ingest.py):
   - Load PDF documents from `docs2/` directory
   - Chunk documents into 700-token chunks with 150-token overlap
   - Generate embeddings using multilingual-e5-base model
   - Upload embeddings to Pinecone vector database
   - Save BM25 keyword search model locally

2. **Query Processing** (main.py):
   - User enters a question
   - **Hybrid Search** combines two retrieval strategies:
     - **Keyword Search**: BM25-based retrieval (local, fast)
     - **Semantic Search**: Vector similarity search (Pinecone, contextual)
   - Results are merged using Reciprocal Rank Fusion (RRF)
   - Cross-Encoder re-ranker refines final ranking
   - Top-5 chunks are retrieved

3. **Answer Generation**:
   - Retrieved chunks provide context
   - Groq LLM (Llama 3.3 70B) generates structured answers
   - LLM is specialized for email deliverability expertise

## Features

- **Hybrid Search**: Combines keyword and semantic search for comprehensive retrieval
- **Re-ranking**: Cross-Encoder model refines search results
- **LLM Integration**: Groq API for fast, accurate answer generation
- **Local Storage**: BM25 model cached locally for efficient keyword search
- **Vector Database**: Pinecone for scalable semantic search
- **Structured Output**: Answers formatted as bullet points with no repetition

## Installation

### Prerequisites

- Python 3.8+
- Pinecone account and API key
- Groq API key
- PDF documents in `docs2/` directory

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd email_rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install langchain langchain-community langchain-pinecone langchain-huggingface langchain-groq
pip install sentence-transformers pypdf pinecone python-dotenv rank-bm25 nltk
```

4. Create a `.env` file with your API keys:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
GROQ_API_KEY=your_groq_api_key
```

5. Place your PDF documents in the `docs2/` directory

## Usage

### Document Ingestion

Run the ingestion pipeline to process documents and build search indices:

```bash
python ingest.py
```

This will:
- Load all PDFs from `docs2/` directory
- Create document chunks
- Generate and upload embeddings to Pinecone
- Create and save BM25 model to `data/bm25.pkl`

### Interactive Query Interface

Start the Q&A system:

```bash
python main.py
```

Enter your questions at the prompt:
```
Ask Question (type 'exit' to quit): How do I improve email deliverability?
```

The system will:
- Retrieve relevant document chunks
- Generate a comprehensive answer
- Display the answer and ranked source chunks

Type 'exit' to quit the application.

## Configuration

Edit `config.py` to customize:

```python
CHUNK_SIZE = 700              # Characters per chunk
CHUNK_OVERLAP = 150           # Overlap between chunks
TOP_K = 5                     # Number of chunks to retrieve
```

## Components Details

### Ingestion Pipeline

- **PDF Loader**: Uses LangChain's PyPDFDirectoryLoader
- **Chunker**: RecursiveCharacterTextSplitter with configurable size and overlap
- **Embedder**: HuggingFace intfloat/multilingual-e5-base model
- **Pinecone Upload**: Stores vectors in Pinecone for semantic search

### Search Strategies

#### Keyword Search (BM25)
- Local, fast retrieval based on term frequency
- NLTK tokenization for preprocessing
- Persisted to `data/bm25.pkl`

#### Semantic Search
- Vector similarity search via Pinecone
- HuggingFace multilingual embeddings
- Captures semantic meaning and context

#### Hybrid Search
- Combines BM25 and semantic results
- Reciprocal Rank Fusion for score aggregation
- Cross-Encoder re-ranking for final refinement
- Returns top 5 results

### LLM Integration

- **Model**: Groq Llama 3.3 70B Versatile
- **Purpose**: Email deliverability expert
- **Features**: 
  - Low temperature (0.1) for consistent results
  - Structured output with bullet points
  - Deduplication and merging of information

## Directory Structure

```
email_rag/
├── docs2/              # PDF documents (input)
├── data/               # Generated data
│   └── bm25.pkl       # Serialized BM25 model
├── ingestion/          # Document processing
├── pro/                # Search implementations
├── llm/                # LLM integration
├── config.py           # Configuration
├── ingest.py           # Ingestion script
├── main.py             # Main application
├── reranker.py         # Re-ranking logic
├── .env                # Environment variables (not committed)
└── README.md           # This file
```

## Dependencies

- **LangChain**: Framework for LLM applications
- **Pinecone**: Vector database for embeddings
- **Groq**: Fast LLM API
- **HuggingFace**: Pre-trained embeddings
- **Sentence Transformers**: Cross-Encoder for re-ranking
- **PyPDF**: PDF document loading
- **rank-bm25**: BM25 keyword search implementation
- **NLTK**: Natural language processing toolkit





