from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document


def store_in_vector_space(docs):
    """Store the scraped content in a vector database."""
    try:
        embeddings = OllamaEmbeddings(model="llama3.1")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
        
        chunks = []
        for doc in docs:
            chunks.extend(text_splitter.split_text(doc.page_content))
        
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        vector_store.save_local('index.faiss')
        
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to store data in vector space: {e}")