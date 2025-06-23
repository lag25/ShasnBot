
# chat_memory.py (formerly langgraph_memory_chain.py)

from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph

from src.prompt import memory_prompt,SAMPLE

class GraphState(TypedDict):
    '''Graph to manage state'''
    messages: Annotated[list, "chat_history"]

def build_langgraph_qa(vectordb, model_name="mistral", k=2, lambd=0.9):
    '''Builds a LangGraph workflow for QA with memory'''
    # Set up retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": lambd})
    if (model_name != "falcon-7b-instruct"):
        llm = Ollama(model=model_name)
    else:
        google_api_key = os.environ.get("GOOGLE_API_KEY")

        if not google_api_key:
            raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")
        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    # Prompt includes history
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SAMPLE),
        MessagesPlaceholder(variable_name="messages"),
        # You could also add a system message if needed here
    ])

    # LLMChain that takes in messages list
    chain = LLMChain(llm=llm, prompt=prompt)

    # Define rag_with_memory node

    def rag_with_memory(state: GraphState) -> GraphState:
        messages = state["messages"]

        # 🧠 Extract latest user query
        latest_question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))

        # 📚 Retrieve relevant documents
        docs = retriever.get_relevant_documents(latest_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # ⚠️ FIX: Add context as a SystemMessage, not HumanMessage
        messages_with_context = messages + [SystemMessage(content=f"Relevant context:\n{context}")]

        # 🧠 Get LLM response
        result = chain.invoke({"messages": messages_with_context})

        # 🧾 Append AI response to the original message list
        return {"messages": messages + [AIMessage(content=result['text'])]}

    # LangGraph setup
    graph = StateGraph(GraphState)
    graph.add_node("rag", rag_with_memory)
    graph.set_entry_point("rag")
    graph.set_finish_point("rag")

    return graph.compile(), retriever
