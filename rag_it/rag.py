from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def rag_chain():
    LM_STUDIO = "http://localhost:1234/v1"

    llm = ChatOpenAI(
        model="lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        openai_api_base=LM_STUDIO,
        openai_api_key="dummy-key",
        temperature=0.85,
        max_tokens=8000,
    )

    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based on the following context.
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions]
        [Instructions] Question: {input}
        Context: {context}
        Answer: [/Instructions]
    """
    )
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(
        persist_directory="./sql_chroma_db", embedding_function=embedding
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    return chain


def ask(query: str):
    chain = rag_chain()
    result = chain.invoke({"input": query})

    print(result["answer"])
    for doc in result["context"]:
        print("Source:", doc.metadata["source"])


ask("what is the data flow architecture?")
