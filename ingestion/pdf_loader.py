from collections import defaultdict
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

def load_documents(folder):
    """
    Load all PDFs in `folder` and merge pages per source file.
    Preserves original PDF loader metadata from the first page while merging.
    """
    loader = PyPDFDirectoryLoader(folder)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    by_source = defaultdict(list)
    for page in pages:
        by_source[page.metadata.get("source", "unknown")].append(page)

    docs = []
    for source, pgs in by_source.items():
        # Pages order sort
        pgs.sort(key=lambda d: d.metadata.get("page", 0))
        
        # Continuous unbroken text string
        full_text = "\n\n".join(p.page_content for p in pgs)
        
        # Pehle page ka PDF loader metadata copy karo
        merged_metadata = pgs[0].metadata.copy() if pgs else {}
        merged_metadata["num_pages"] = len(pgs)
        
        docs.append(
            Document(
                page_content=full_text,
                metadata=merged_metadata,
            )
        )

    print(f"Merged into {len(docs)} documents (one per source file)")
    return docs