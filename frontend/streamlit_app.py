import sys
import os
import base64
import streamlit as st
from highlight_chunks_in_pdf import highlight_chunks_in_pdf
from src.prompt import qa_prompt
print("PID:", os.getpid())


# --- Setup for Modular Project Import ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

pdf_input = os.path.join(current_dir, "sample_rulebook.pdf")
pdf_output = os.path.join(current_dir, "highlighted_rulebook.pdf")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import your RAG logic ---
from src.qa import build_qa_chain, answer_query

# --- Streamlit UI Setup ---
db_name = st.selectbox("Choose vectorstore backend", ["chroma", "faiss"])

@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

@st.cache_resource
def get_cached_vectorstore(db_name, embeddings):
    return load_vectorstore(db_name=db_name, embeddings=embeddings)

# 🔁 Create embeddings once, pass into vectorstore loader
embeddings = get_embeddings()
vectordb = get_cached_vectorstore(db_name, embeddings)
st.set_page_config(page_title="Rulebook QA Chatbot", layout="wide")
st.title("📘 Rulebook Chat Assistant")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Define your RAG response logic ---
def get_rag_response(query: str) -> tuple[str, list[dict]]:
    """
    Calls your RAG pipeline and returns the response and relevant chunks.
    """
    qa_chain,retriever = build_qa_chain()  # Can change Model and DB
    rag_answer,retrieved_chunks = answer_query(qa_chain,retriever, query)

    # Placeholder for testing - replace dummy_chunks with real retrieval later
    highlight_chunks_in_pdf(pdf_input,retrieved_chunks,output_path=pdf_output)

    return rag_answer["result"], retrieved_chunks


# --- Display Chat History ---
for user_q, bot_a in st.session_state.chat_history:
    st.chat_message("user").markdown(user_q)
    st.chat_message("assistant").markdown(bot_a)

# --- Chat Input ---
query = st.chat_input("Ask something about the rulebook...")

if query:
    st.chat_message("user").markdown(query)

    with st.spinner("Answering..."):
        response_text, retrieved_chunks = get_rag_response(query)

    st.chat_message("assistant").markdown(response_text)
    st.session_state.chat_history.append((query, response_text))

    # --- Show Retrieved Chunks ---
    with st.expander("📖 Retrieved Rulebook Context"):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}** — Page {chunk.metadata.get('page', 'N/A')}, Section: {chunk.metadata.get('section', 'N/A')}")
            st.code(chunk.page_content.strip(), language="markdown")

    # --- Show Highlighted PDF Download ---
    if os.path.exists(pdf_output):
        with open(pdf_output, "rb") as f:
            st.download_button("⬇️ Download Highlighted PDF", f, file_name="highlighted_rulebook.pdf")
            st.info("Highlighted rulebook shows exactly where the answer came from.")
    else:
        st.warning("⚠️ Could not generate highlighted PDF.")
