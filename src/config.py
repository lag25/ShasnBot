import os

PARENT_PATH = os.path.abspath(os.path.dirname(__file__))
INDEX_PATH = os.path.join(PARENT_PATH, 'vectordb')
FILE_PATH = os.path.join(PARENT_PATH, 'data', 'rulebook.pdf')
CHROMA_PATH = os.path.join(INDEX_PATH,'chromaDB')
FAISS_PATH = os.path.join(INDEX_PATH,'faissDB')
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"