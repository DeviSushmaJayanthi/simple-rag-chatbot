import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import streamlit as st

# 1. SETUP & CONFIGURATION
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.sidebar.warning("API Key notfound in environment")
    user_key = st.sidebar.text_input("Enter your Google API Key",type = "password")
    selected_api_key = user_key
else:
    st.sidebar.success("API key loaded from .env")
    selected_api_key = GOOGLE_API_KEY


# Define Paths
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# 2. INITIALIZE MODELS (Using 2026 Stable Models)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=selected_api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=selected_api_key,
    temperature=0.2  # Keep it factual and concise
)

def build_vector_store():
    """Processes PDFs and creates a local vector database."""
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_splits = text_splitter.split_documents(docs)
    
    vector_db = FAISS.from_documents(final_splits, embeddings)
    
    # Save the index so we don't have to re-process next time
    os.makedirs("vectorstore", exist_ok=True)
    vector_db.save_local(DB_FAISS_PATH)
    return vector_db

def get_rag_response(user_input):
    """Retrieves context and generates a natural language answer."""
    
    # Check if DB exists; if not, create it
    if not os.path.exists(DB_FAISS_PATH):
        vector_db = build_vector_store()
    else:
        vector_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # 3. DEFINE THE PROMPT (System Instructions)
    system_prompt = (
        "You are an expert assistant. Use the provided context to answer the user's question. "
        "If the answer isn't in the context, say you don't know based on the provided files. "
        "Keep the answer well-structured and professional.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        ("human", "{input}"),
    ])

    # 4. CREATE THE CHAIN
    # Create the 'stuffing' chain (how documents are formatted for the prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain (how documents are fetched from FAISS)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. INVOKE
    response = rag_chain.invoke({"input": user_input})
    return response

if __name__ == "__main__":
    
    st.title("Question Answering System")
    
    query = st.text_input("Enter your query..")
    
    # Get the complete response dictionary
    if st.button("Get Response"):
        if not selected_api_key:
            st.error("please provide the google api key to proceed further")
        else:

            result = get_rag_response(query)
            st.success(result["answer"])

            # Output which files it looked at
            st.write("SOURCES")
            sources = {os.path.basename(doc.metadata['source']) for doc in result["context"]}
            for source in sources:
                st.write(f"- {source}")
