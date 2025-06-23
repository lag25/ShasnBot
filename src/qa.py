import os
import time
from typing import Generator
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from streamlit import session_state


from .vectorstore import load_vectorstore, check_for_vectorstore
from .sql_logs import log_rag_event

from .prompt import qb_prompt,memory_prompt
from .utils import setup_logger


logger = setup_logger(log_to_file=True)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI  # or Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_qa_chain_memory(
    vectordb,
    prompt,
    model_name="Mistral",
    db_name='chroma',
    k=2,
    lambd=0.99,
    memory=None
):
    '''Build a QA chain with memory'''
    logger.info(f"Building Conversational QA chain with {model_name}")


    # 🔁 LLM instantiation (if not passed)

    # 📚 MMR Retriever
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": lambd}
    )
    llm = Ollama(model=model_name)

    # 🔗 Conversational RAG Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain


def build_qa_chain(vectordb,prompt=qb_prompt, model_name="Mistral", db_name='chroma',k=2,lambd=0.99,Ollama_use=True):
    '''Returns a QA-Chain (LangChain object) along with a retriever. Usually once called during state changes'''
    logger.info(f"Building QA chain with {model_name}")
    if(model_name!="falcon-7b-instruct"):
        llm = Ollama(model=model_name)
    else:
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = "AIzaSyDBMa_0v2me-OT5tr-yToNb6ttT3JcdpPw" # got to change this CANNOT have api key in the open like that

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
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

def get_relevant_docs(query:str,retriever):
    '''Retrieves documents from a vectorstore. Currently not used but will be for a cleaner and modular approach'''
    return retriever.invoke(query)

def stream_and_collect(generator: Generator[str, None, None]):
    '''A generator wrapper to run my llm response generator and also get response in list to append in chat'''
    collected_tokens = []
    def wrapped_generator():
        for token in generator:
            collected_tokens.append(token)
            yield token

    return wrapped_generator(), collected_tokens

def answer_query_stream_to_ui(qa_chain, retriever, query: str,log_sql=True):
    """
    Stream LLM output using a generator (no memory)
    """
    logger.info(f"Running streamed query: {query}")
    start_query_time = time.time()
    stream_iter = qa_chain.stream({"query": query})


    for chunk in stream_iter:
        token = chunk.get("result", "") or chunk.get("answer", "")
        yield token
    start_query_time = time.time() - start_query_time
    logger.info(f"Query completed in {start_query_time:.2f} seconds")
    if(log_sql):   # Push event to sqlite
        log_rag_event(start_query_time,query,session_state.active_model,session_state.active_store)

def answer_query(qa_chain, retriever, query: str):
    '''Naive implementation. Leaving for basic testing'''
    logger.info(f"Running query: {query}")
    start_query_time = time.time()
    result = qa_chain.invoke(query)
    logger.info(f"Query succesfully executed in {time.time() - start_query_time}")
    return result, retriever.get_relevant_documents(query)



def answer_query_langgraph(graph_app, query: str) -> str:
    '''Adds the query to the graph state and invokes llm with memory'''
    session_state.graph_state["messages"].append(HumanMessage(content=query))
    length_graph_state = len(session_state.graph_state)
    start_query_time = time.time()
    session_state.graph_state = graph_app.invoke(session_state.graph_state)
    start_query_time = time.time() - start_query_time
    logger.info(f"Query successfully executed in {start_query_time} seconds")
    return session_state.graph_state["messages"][-1].content

if __name__ == "__main__":

    start = time.time()
    qa_chain, retriever = build_qa_chain()
    print(answer_query(qa_chain, retriever, query="What are volatile areas ?"))
    logger.info(f"Query took {time.time() - start:.2f} seconds to execute")
