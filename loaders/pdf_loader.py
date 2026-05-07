# PDF Loader module
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    texts = []
    for page in pages:
        texts.append(page.page_content)
    
    return texts   # return list (important)