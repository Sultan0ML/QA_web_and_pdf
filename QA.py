import asyncio
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
import streamlit as st
from langchain_core.documents import Document
import os

os.system("playwright install")  # Make sure the necessary dependencies are installed
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def scrap_data(url):
    """Scrape data from the given URL."""
    loaders = AsyncChromiumLoader([url],user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    docs = loaders.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["p"])

    # Return a list of Document objects
    return docs_transformed[0].page_content

def store_in_vector_space(docs):
    """Store the scraped content in a vector database."""
    embeddings = OllamaEmbeddings(model="llama3.1")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))

    # Wrap each chunk into a Document object
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create FAISS DB
    vectorStore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorStore.save_local('faiss.index')
    
    return vectorStore

def ask_question(vector_db, query):
    """Answer user questions based on the vector database."""
    llm = OllamaLLM(model="llama3.1", temperature=0.5)

    # Search the vector database for relevant chunks
    results = vector_db.similarity_search(query, k=3)

    # Concatenate the chunks into context
    context = "\n".join([result.text for result in results])

    # Create a prompt and get the answer from the LLM
    prompt_template = PromptTemplate(template="""
    Use the following context to answer the question:
    {context}

    Question: {question}
    Answer:
    """)

    prompt = prompt_template.format(context=context, question=query)
    response = llm(prompt)

    return response

# Streamlit App
st.title("Ask Questions from a Web Article")

# Move URL input to sidebar
url = st.sidebar.text_input("Enter article URL")

if url:
    with st.spinner("Scraping data from the URL..."):
        try:
            docs = scrap_data(url)
        except Exception as e:
            st.error(f"Error scraping data: {e}")
            docs = []

    if docs:
        st.success("Data scraped successfully!")
        
        # Store in vector space
        with st.spinner("Storing data in vector space..."):
            try:
                vector_db = store_in_vector_space(docs)
                st.success("Data stored in vector space!")
            except Exception as e:
                st.error(f"Error storing data in vector space: {e}")
                vector_db = None
    if vector_db is not None:
        # User can ask questions
        question = st.text_input("Ask a question about the article")

        if question and vector_db:
            with st.spinner("Generating response..."):
                try:
                    answer = ask_question(vector_db, question)
                    st.text_area("Answer", answer, height=200)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.error("Failed to scrape the content. Please check the URL.")
