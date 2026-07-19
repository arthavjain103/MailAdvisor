from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(folder):

    loader = PyPDFDirectoryLoader(folder)

    docs = loader.load()

    print(f"Loaded {len(docs)} pages")

    return docs