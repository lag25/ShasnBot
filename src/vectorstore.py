import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.documents import Document

from .config import INDEX_PATH, FILE_PATH, CHROMA_PATH, FAISS_PATH, MODEL_NAME    # Gets vectorDB directory
from .loader import create_chunks

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}



def build_chroma_index(chunks: list[Document]):
    '''Builds an instance of Chroma VectorDB'''
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    vectorstore = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory = CHROMA_PATH
    )
    #print(f"Saved {len(chunks)} chunks to {CHROMA_FILE}")
    return vectorstore

def build_faiss_index(docs: list[Document]):
    '''Builds an instance of FAISS VectorDB'''
    #model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore

def check_for_vectorstore(db_name:str):
    '''Checks if a vectorDB already exists'''
    if(db_name=='chroma'):
        db_path = CHROMA_PATH
    else:
        db_path = FAISS_PATH
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"An instance of {db_name} found.")
        return True
    else:
        raise FileNotFoundError(f"No ChromaDB found at {CHROMA_PATH}. Please build an instance")

def load_vectorstore(embeddings,db_name:str='chroma'):
    '''Returns the instance of a vectorDB (FAISS or Chroma for now)'''
    if(db_name == "chroma"):
        return Chroma(persist_directory=os.path.join(INDEX_PATH,'chromaDB'), embedding_function=embeddings)
    else:
        return FAISS.load_local(os.path.join(INDEX_PATH,'faissDB'), embeddings, allow_dangerous_deserialization=True)

if(__name__ == "__main__"):
    '''Usage to create/check for stores'''
    try:
        import asyncio
        from .loader import create_docs
        docs = asyncio.run(create_docs())
        docs = create_chunks(docs)
        #model_name = "sentence-transformers/all-mpnet-base-v2"
        #embeddings = HuggingFaceEmbeddings(model_name=model_name)
        #print(load_vectorstore(),embeddings)
        build_faiss_index(docs)
        print(check_for_vectorstore('faiss'))
        print("RUN SUCCESSFUL")
    except Exception as e:
        print(e)

