import argparse
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings


def rag_chain(query_text: str):
    LM_STUDIO = "http://localhost:1234/v1"

    llm = ChatOpenAI(
        model="lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        openai_api_base=LM_STUDIO,
        openai_api_key="dummy-key",
        temperature=0.85,
        max_tokens=8000,
    )

    LM_STUDIO = "http://localhost:1234/v1"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v1.5",
        openai_api_base=LM_STUDIO,
        openai_api_key="dummy-key",
        check_embedding_ctx_length=False,
    )

    db = Chroma(persist_directory="./sql_chroma_db", embedding_function=embeddings)

    results = db.similarity_search_with_score(query_text, k=5)

    # retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 3, "score_threshold": 0.5},
    # )
    # document_chain = create_stuff_documents_chain(llm, prompt)
    # chain = create_retrieval_chain(retriever, document_chain)
    #
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based on the following context.
        If you don't know the answer, then reply, No Context available for this question {question}. [/Instructions]
        [Instructions] Question: {question}
        Context: {context}
        Answer: [/Instructions]
    """
    )

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt = prompt.format(context=context_text, question=query_text)
    response_text = llm.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def ask(query: str):
    response_text = rag_chain(query)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    ask(query_text)


if __name__ == "__main__":
    main()
