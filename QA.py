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

# Step 2: Store the article in vector space
def store_in_vector_space(docs):
    """Store the scraped content in a vector database."""
    try:
        embeddings = OllamaEmbeddings(model="llama3.1")

        # Split the text into smaller chunks to preserve more context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)  # Smaller chunks
        chunks = []
        for doc in docs:
            chunks.extend(text_splitter.split_text(doc.page_content))

        # Wrap each chunk into a Document object
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Create FAISS DB
        vectorStore = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectorStore.save_local('faiss.index')
        
        # Print stored documents for debugging
        print("Stored documents in the vector store:")
        for doc in documents[:5]:  # Print the first 5 documents for debugging
            print(doc.page_content)

        return vectorStore

    except Exception as e:
        print(f"Error storing data in vector space: {e}")
        return None


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

st.title("Ask Questions from a Web Article")

# Move URL input to sidebar
url = st.sidebar.text_input("Enter article URL")

# Check if the document and vector database already exist in session state
if 'docs' not in st.session_state or 'vector_db' not in st.session_state:
    if url:
        with st.spinner("Scraping data from the URL..."):
            try:
                docs = scrap_data(url)  # Ensure scrap_data function is defined elsewhere
                docs = [Document(page_content=docs)]  # Create Document instance (ensure docs is the correct content)
                st.session_state.docs = docs  # Save docs in session state
            except Exception as e:
                st.error(f"Error scraping data: {e}")
                docs = []

        if docs:
            st.success("Data scraped successfully!")
            
            # Store in vector space
            with st.spinner("Storing data in vector space..."):
                try:
                    vector_db = store_in_vector_space(docs)  # Ensure store_in_vector_space is defined
                    st.session_state.vector_db = vector_db  # Save vector_db in session state
                    st.success("Data stored in vector space!")
                except Exception as e:
                    st.error(f"Error storing data in vector space: {e}")
                    vector_db = None

else:
    docs = st.session_state.docs
    vector_db = st.session_state.vector_db

# User can ask questions if the data is available
if docs and vector_db:
    question = st.text_input("Ask a question about the article")

    if question:
        with st.spinner("Generating response..."):
            try:
                answer = ask_question(vector_db, question)  # Ensure ask_question is defined
                st.text_area("Answer", answer, height=200)
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    st.error("Failed to scrape or store the content. Please check the URL.")