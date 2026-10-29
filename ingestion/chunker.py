import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Normal / Fallback Splitter baki sabhi non-email files ke liye
basic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def split_email_templates(doc):
    """
    Structural chunking for email.pdf:
    - 1 Template = 1 Complete Chunk
    - 🔍 AI Alerts separated into 'embedded_annotation'
    - Preserves all original PDF loader metadata
    """
    text = doc.page_content
    base_metadata = doc.metadata.copy()

    parts = re.split(r"(?=Template \d+:)", text)
    chunks = []

    for part in parts:
        stripped = part.strip()
        if not stripped or not stripped.startswith("Template"):
            continue

        name_match = re.match(r"Template (\d+): (.+)", stripped)
        alert_match = re.search(r"🔍.*", part, re.DOTALL)
        body = re.sub(r"🔍.*", "", part, flags=re.DOTALL).strip()

        # 1. Main Template Chunk
        if body:
            meta = base_metadata.copy()
            meta.update({
                "chunk_type": "template",
                "template_num": name_match.group(1) if name_match else None,
                "template_name": name_match.group(2) if name_match else None,
            })
            chunks.append(Document(page_content=body, metadata=meta))

        # 2. Alert/Annotation Chunk
        if alert_match:
            meta = base_metadata.copy()
            meta.update({"chunk_type": "embedded_annotation"})
            chunks.append(Document(page_content=alert_match.group().strip(), metadata=meta))

    return chunks

def chunk_documents(documents):
    """
    Conditional Routing:
    - If document is 'email.pdf' -> Structural Template Chunking
    - All other documents      -> Basic Recursive Character Chunking
    """
    all_chunks = []

    for doc in documents:
        source = doc.metadata.get("source", "")

        if "email" in source.lower():
            # Structural Chunking for email templates
            chunks = split_email_templates(doc)
        else:
            # Basic Chunking for all other PDFs
            chunks = basic_splitter.split_documents([doc])

        all_chunks.extend(chunks)

    print(f"Total Chunks Created: {len(all_chunks)}")
    return all_chunks