import os
import time

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain

from .vectorstore import load_vectorstore, check_for_vectorstore

from .prompt import qc_prompt
from .utils import setup_logger


logger = setup_logger(log_to_file=True)


def build_qa_chain(vectordb,prompt=qc_prompt, model_name="Mistral", db_name='chroma',k=2,lambd=0.99):
    #ToDo : make vectorDB a parameters of this function along with llm
    logger.info(f"Building QA chain with {model_name}")

    # 🔽 Lazy imports for heavy modules



    #if check_for_vectorstore(db_name):
     #   vectordb = load_vectorstore(db_name)

    llm = Ollama(model=model_name)
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": lambd}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    ), retriever


def answer_query_stream_to_ui(qa_chain, retriever, query: str, ui_placeholder=None):
    """
    Stream LLM output token by token, optionally live-updating a Streamlit placeholder.
    """
    logger.info(f"Running streamed query: {query}")
    start_query_time = time.time()

    full_response = ""
    stream_iter = qa_chain.stream({"query": query})

    for chunk in stream_iter:
        token = chunk.get("result", "") or chunk.get("answer", "")
        full_response += token

        if ui_placeholder:
            ui_placeholder.markdown(full_response + "▌")

    if ui_placeholder:
        ui_placeholder.markdown(full_response)

    logger.info(f"Query completed in {time.time() - start_query_time:.2f} seconds")
    return full_response, retriever.get_relevant_documents(query)


def answer_query_stream_to_ui_with_memory(qa_chain, retriever, query: str, ui_placeholder=None):
    """
    Stream LLM output token by token, optionally live-updating a Streamlit placeholder.
    Works for both standard QA and ConversationalRetrievalChain (memory).
    """
    logger.info(f"Running streamed query: {query}")
    start_query_time = time.time()

    full_response = ""

    # Detect whether chain uses memory
    input_key = "question" if isinstance(qa_chain, ConversationalRetrievalChain) else "query"
    stream_iter = qa_chain.stream({input_key: query})

    try:
        for chunk in stream_iter:
            token = chunk.get("result", "") or chunk.get("answer", "")
            full_response += token

            if ui_placeholder:
                ui_placeholder.markdown(full_response + "▌")
    except Exception as e:
        logger.error(f"❌ Streaming failed: {e}")
        raise e

    if ui_placeholder:
        ui_placeholder.markdown(full_response)

    logger.info(f"✅ Query completed in {time.time() - start_query_time:.2f} seconds")
    return full_response, retriever.get_relevant_documents(query)


def answer_query_stream(qa_chain, retriever, query: str):
    logger.info(f"Running query: {query}")
    start_query_time = time.time()

    stream_iter = qa_chain.stream({"query": query})  # Dict-style input is required
    full_response = ""

    for chunk in stream_iter:
        token = chunk.get("result", "") or chunk.get("answer", "")
        print(token, end="", flush=True)
        full_response += token

    logger.info(f"\nQuery successfully executed in {time.time() - start_query_time:.2f} seconds")
    return full_response, retriever.get_relevant_documents(query)


def answer_query(qa_chain, retriever, query: str):
    logger.info(f"Running query: {query}")
    start_query_time = time.time()
    result = qa_chain.invoke(query)
    logger.info(f"Query succesfully executed in {time.time() - start_query_time}")
    return result, retriever.get_relevant_documents(query)


if __name__ == "__main__":
    start = time.time()
    qa_chain, retriever = build_qa_chain()
    print(answer_query(qa_chain, retriever, query="What are volatile areas ?"))
    logger.info(f"Query took {time.time() - start:.2f} seconds to execute")
