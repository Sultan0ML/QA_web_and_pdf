import streamlit as st
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb

import os

# Install necessary dependencies for Playwright
os.system("playwright install")

# Ensure FAISS compatibility
os.environ['FAISS_NO_AVX2'] = '1'
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")

# Initialize session state
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

# Step 1: Scrape data from the URL
def scrap_data(url):
    """Scrape data from the given URL."""
    try:
        loaders = AsyncChromiumLoader(
            [url],
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        docs = loaders.load()
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["p"])
        return docs_transformed[0].page_content
    except Exception as e:
        raise RuntimeError(f"Failed to scrape data: {e}")

# Step 2: Store the article in vector space


# Step 2: Store the article in vector space


# def store_in_vector_space(docs):

#     os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
#     embeddings=OllamaEmbeddings(model="llama3.1")
#     pc = Pinecone()
#     index = pc.Index("web-pdf")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
        
#         # Split the documents into chunks
#     chunks = []
#     for doc in docs:
#         chunks.extend(text_splitter.split_text(doc.page_content))
        
#         # Create Document objects
#     documents = [Document(page_content=chunk) for chunk in chunks]

#     vector_store = PineconeVectorStore(embedding=embeddings, index=index)
#     for document in documents:
#         vector_store.add_texts([document.page_content])
#     return vector_store

def store_in_vector_space(docs):
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Use or create a collection in ChromaDB
    collection_name = "chroma_docs"
    collection = client.get_or_create_collection(name=collection_name)
    
    # Use the embedding model (assuming OllamaEmbeddings is compatible)
    embeddings = OllamaEmbeddings(model="llama3.1")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))
    
    # Prepare data for ChromaDB insertion
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Store embeddings and documents in ChromaDB
    for doc in documents:
        # Generate embedding for each document chunk
        vector = embeddings.embed(doc.page_content)
        # Add the document with the embedding to the ChromaDB collection
        collection.add(
            documents=[doc.page_content], 
            embeddings=[vector], 
            ids=[str(hash(doc.page_content))]
        )
        
    return collection


# Step 3: Answer questions based on vector database
def ask_question(vector_db, query):
    """Answer user questions based on the vector database."""
    try:
        llm = OllamaLLM(model="llama3.1", temperature=0.5)
        results = vector_db.similarity_search(query, k=3)
        context = "\n".join([result.page_content for result in results])

        prompt_template = PromptTemplate(template="""
        Use the following context to answer the question:
        {context}

        Question: {question}
        Answer:
        """)
        prompt = prompt_template.format(context=context, question=query)
        response = llm(prompt)
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to generate answer: {e}")

# Streamlit app
st.title("Ask Questions from a Web Article")

# Move URL input to sidebar
url = st.sidebar.text_input("Enter article URL", help="Provide the URL of the article you want to analyze.")

# Check if the document and vector database already exist
if st.session_state.docs is None or st.session_state.vector_db is None:
    if url:
        with st.spinner("Scraping data from the URL..."):
            try:
                docs = scrap_data(url)
                docs = [Document(page_content=docs)]
                st.session_state.docs = docs
                st.success("Data scraped successfully!")
                st.write(docs)
            except Exception as e:
                st.error(f"Error scraping data: {e}")
                docs = []

        if docs:
            with st.spinner("Storing data in vector space..."):
                try:
                    vector_db = store_in_vector_space(docs)
                    st.session_state.vector_db = vector_db
                    st.success("Data stored in vector space!")
                except Exception as e:
                    st.error(f"Error storing data in vector space: {e}")

else:
    docs = st.session_state.docs
    vector_db = st.session_state.vector_db

# Allow the user to ask questions
if st.session_state.docs and st.session_state.vector_db:
    question = st.text_input("Ask a question about the article", help="Type your question here.")
    if question:
        with st.spinner("Generating response..."):
            try:
                answer = ask_question(vector_db, question)
                st.text_area("Answer", answer, height=200)
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    st.error("Failed to scrape or store the content. Please check the URL.")
