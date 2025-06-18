import os
import time

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from .vectorstore import load_vectorstore, check_for_vectorstore

from .prompt import qa_prompt
from .utils import setup_logger


logger = setup_logger(log_to_file=True)


def build_qa_chain(vectordb,prompt=qa_prompt, model_name="Gemma3", db_name='chroma',k=2,lambd=0.99):
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
