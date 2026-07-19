from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore

from config import PINECONE_API_KEY, PINECONE_INDEX

from ingestion.embedder import get_embedding_model


def upload(chunks):

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index = pc.Index(PINECONE_INDEX)

    embeddings = get_embedding_model()

    PineconeVectorStore.from_documents(

        chunks,

        embedding=embeddings,

        index_name=PINECONE_INDEX

    )

    print("Uploaded Successfully")