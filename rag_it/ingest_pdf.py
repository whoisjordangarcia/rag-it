from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "./sql_chroma_db"


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/things.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    LM_STUDIO = "http://localhost:1234/v1"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v1.5",
        openai_api_base=LM_STUDIO,
        openai_api_key="dummy-key",
        check_embedding_ctx_length=False,
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("PDF has been ingested")
    else:
        print("no new documents to add")


def ingest():
    loader = PyPDFLoader("~/Downloads/google_file_system_whitepaper.pdf")
    documents = loader.load_and_split()
    chunks = split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    print("no new documents to add")

    add_to_chroma(chunks)


ingest()
