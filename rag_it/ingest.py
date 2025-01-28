from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core import embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest():
    loader = PyPDFLoader("~/Downloads/google_file_system_whitepaper.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks.")

    embedding = FastEmbedEmbeddings()
    Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory="./sql_chroma_db"
    )
    print("PDF has been ingested")


ingest()
