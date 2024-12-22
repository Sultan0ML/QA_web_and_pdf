from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st

def scrap_data(url):
    """Scrape data from the given URL."""
    loaders = AsyncChromiumLoader([url], user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    docs = loaders.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["p"])

    return docs_transformed[0].page_content if docs_transformed else ""

def store_in_vector_space(content):
    """Store the scraped content in a vector database."""
    embeddings = OllamaEmbeddings(model="llama3.1")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(content)

    # Create Chroma DB
    chroma_db = Chroma(persist_directory="chroma_store", embedding_function=embeddings)
    
    for i, chunk in enumerate(chunks):
        chroma_db.add_texts([chunk], metadatas=[{"chunk_index": i}])
    
    return chroma_db

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

# Input URL
url = st.text_input("Enter article URL")

if url:
    with st.spinner("Scraping data from the URL..."):
        content = scrap_data(url)

    if content:
        st.success("Data scraped successfully!")
        
        # Store in vector space
        with st.spinner("Storing data in vector space..."):
            vector_db = store_in_vector_space(content)

        st.success("Data stored in vector space!")

        # User can ask questions
        question = st.text_input("Ask a question about the article")

        if question:
            with st.spinner("Generating response..."):
                answer = ask_question(vector_db, question)

            st.text_area("Answer", answer, height=200)

    else:
        st.error("Failed to scrape the content. Please check the URL.")

