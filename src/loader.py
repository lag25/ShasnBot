import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter

async def create_docs(file_path:str=r"D:\Desktop\Github for resume\LatticeBuild_Project\data\prem_rules.pdf") -> list:
    '''Creates a LangChain Document object for the pages in pdf doc.
    Async func so I have to run this with asyncio.run()'''

    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def create_chunks(docs,chunk_size=500,chunk_overlap=100):
    '''Creates chunk out of Documents'''
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    return chunks








