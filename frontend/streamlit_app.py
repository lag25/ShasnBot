import sys
import os
import subprocess
import requests


import streamlit as st
import base64

from highlight_chunks_in_pdf import highlight_chunks_in_pdf


# Debug process ID
print("PID:", os.getpid())

# --- Setup for Modular Project Import ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import core logic ---
from src.qa import build_qa_chain, answer_query
from src.vectorstore import load_vectorstore
from src.config import MODEL_NAME  # default model (e.g., mistral)

# --- Function to check active Ollama model and processor use ---
import subprocess
import requests
import time


def get_ollama_gpu_status(selected_model: str, ensure_gpu: bool = False):
    import subprocess
    import requests
    import time

    try:
        r = requests.get("http://localhost:11434/api/ps")
        if not r.ok:
            return "🔴 Could not connect to Ollama API", None, None

        data = r.json()
        models = data.get("models", [])

        if not isinstance(models, list):
            return f"🔴 Unexpected format (not a list): {models}", None, None

        if not models:
            if ensure_gpu:
                subprocess.Popen(["ollama", "run", selected_model, "--keepalive", "30m"])
                return f"✅ {selected_model} is now launching (no models were running)", selected_model, "Launching"
            return "🟡 No active models running", None, None

        # Check if the selected model is already running
        for m in models:
            model_name = m.get("name", "").lower()
            processor = m.get("details", {}).get("quantization_level", "Unknown quantization")
            if selected_model.lower() in model_name:
                return (
                    f'🟢 **{m["name"]}** is already running ({processor})',
                    m["name"],
                    processor
                )

        # If ensure_gpu is True, stop others and launch selected model
        if ensure_gpu:
            for m in models:
                name = m.get("name")
                if name:
                    requests.post("http://localhost:11434/api/stop", json={"name": name})
                    time.sleep(1)

            subprocess.Popen(["ollama", "run", selected_model, "--keepalive", "30m"])
            return f"✅ {selected_model} is now launching (previous models stopped)", selected_model, "Launching"

        return f"🟠 Another model is running (not `{selected_model}`)", None, None

    except Exception as e:
        return f"🔴 Ollama error: {e}", None, None



# --- Constants ---
pdf_input = os.path.join(current_dir, "sample_rulebook.pdf")
pdf_output = os.path.join(current_dir, "highlighted_rulebook.pdf")
AVAILABLE_MODELS = ["mistral", "Gemma3","phi","gemma3:4b-it-qat","qwen2.5:latest","deepseek-r1:7b","smollm2:1.7b"]
AVAILABLE_STORES = ["chroma", "faiss"]

# --- Cached embeddings ---
@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

# --- App Layout ---
st.set_page_config(page_title="Rulebook QA Chatbot", layout="wide")
st.title("📘 Rulebook Chat Assistant")

# --- Sidebar Model + Vectorstore Controls ---
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose LLM Model", AVAILABLE_MODELS, index=0)
selected_store = st.sidebar.selectbox("Choose Vector Store", AVAILABLE_STORES, index=0)


# --- Sidebar Status Output ---
if st.sidebar.button("🔄 Force Selected Model on GPU"):
    status_msg, running_model, proc_type = get_ollama_gpu_status(selected_model, ensure_gpu=True)
    st.sidebar.success(status_msg)

# --- Always show current model status ---
status_msg, running_model, proc_type = get_ollama_gpu_status(selected_model)
st.sidebar.markdown("---")
st.sidebar.markdown(f"🖥️ **Ollama Status:**\n\n{status_msg}")



# --- Init Embeddings ---
embeddings = get_embeddings()

# --- Session state: track model & vectorstore ---
if "active_model" not in st.session_state:
    st.session_state.active_model = None
    st.session_state.active_store = None
    st.session_state.qa_chain = None
    st.session_state.retriever = None
    st.session_state.chat_history = []

# --- Only reload chain if config changes ---
if (st.session_state.active_model != selected_model) or (st.session_state.active_store != selected_store):
    with st.spinner("Loading model and vector store..."):
        vectordb = load_vectorstore(embeddings, selected_store)
        qa_chain, retriever = build_qa_chain(vectordb, model_name=selected_model)

        st.session_state.qa_chain = qa_chain
        st.session_state.retriever = retriever
        st.session_state.active_model = selected_model
        st.session_state.active_store = selected_store
        st.success(f"🔄 Model: {selected_model}, Store: {selected_store} loaded.")

# --- Define core query logic ---
def get_rag_response(query: str) -> tuple[str, list]:
    answer, retrieved_chunks = answer_query(st.session_state.qa_chain, st.session_state.retriever, query)
    #highlight_chunks_in_pdf(pdf_input, retrieved_chunks, output_path=pdf_output)
    return answer["result"], retrieved_chunks

# --- Display chat history ---
for user_q, bot_a in st.session_state.chat_history:
    st.chat_message("user").markdown(user_q)
    st.chat_message("assistant").markdown(bot_a)

# --- Chat input ---
query = st.chat_input("Ask a question about the rulebook...")

if query:
    st.chat_message("user").markdown(query)

    with st.spinner("Answering..."):
        response_text, retrieved_chunks = get_rag_response(query)

    st.chat_message("assistant").markdown(response_text)
    st.session_state.chat_history.append((query, response_text))

    # --- Retrieved context display ---
    with st.expander("📖 Retrieved Rulebook Context"):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}** — Page {chunk.metadata.get('page', 'N/A')}, Section: {chunk.metadata.get('section', 'N/A')}")
            st.code(chunk.page_content.strip(), language="markdown")

    # --- PDF download section ---
    if os.path.exists(pdf_output):
        with open(pdf_output, "rb") as f:
            st.download_button("⬇️ Download Highlighted PDF", f, file_name="highlighted_rulebook.pdf")
            st.info("Highlighted rulebook shows exactly where the answer came from.")
    else:
        st.warning("⚠️ Could not generate highlighted PDF.")
