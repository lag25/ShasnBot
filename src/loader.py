import asyncio
from langchain_community.document_loaders import PyPDFLoader

async def create_docs(file_path:str=r"D:\Desktop\Github for resume\LatticeBuild_Project\data\rulebook.pdf") -> list:
    '''Async func so remember to run with asyncio.run()'''
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages



