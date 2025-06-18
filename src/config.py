import os

PARENT_PATH =  r"D:\Desktop\Github for resume\LatticeBuild_Project"
INDEX_PATH = r"D:\Desktop\Github for resume\LatticeBuild_Project\vectordb"
FILE_PATH = r"D:\Desktop\Github for resume\LatticeBuild_Project\data\rulebook.pdf"
CHROMA_PATH = os.path.join(INDEX_PATH,'chromaDB')
FAISS_PATH = os.path.join(INDEX_PATH,'faissDB')
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"