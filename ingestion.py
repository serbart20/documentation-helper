from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


INDEX_NAME = "langchain-doc-index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader(
        # "langchain-docs/api.python.langchain.com/en/latest/chains"
        "langchain-docs/langchain.readthedocs.io/en/latest/",  encoding="utf-8"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into {len(documents)} chunks")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    # PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    # print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
