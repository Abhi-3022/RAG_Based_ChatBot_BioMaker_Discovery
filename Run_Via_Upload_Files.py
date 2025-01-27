import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import uuid


GOOGLE_API_KEY = 'AIzaSyCigDCyJep9exItOgnnSKUoL22-c7uekbE'
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """


    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def process_user_question(user_question):
    """Process user question and retrieve answer from PDF context."""
    
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDF files first!")
        return "Error: No PDF processed"

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        
        docs = new_db.similarity_search(user_question)
        
        
        chain = get_conversational_chain()
        
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        return response['output_text']
        
    except Exception as e:
        return f"An error occurred: {e}"

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

def display_chat_messages():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    """Main Streamlit application function."""
    
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📄",
        layout="centered"
    )

    
    initialize_session_state()

    
    st.title("📄 PDF Chat Assistant")
    st.caption("Upload a PDF and start chatting with its content!")

    
    with st.sidebar:
        st.header("📤 PDF Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload one or multiple PDF files to analyze"
        )
        
        if st.button("Process PDFs", type="primary"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    
                    raw_text = get_pdf_text(pdf_docs)
                    
                    
                    text_chunks = get_text_chunks(raw_text)
                    
                    
                    get_vector_store(text_chunks)
                    
                    
                    st.session_state.pdf_processed = True
                    
                    
                    st.success("PDFs processed successfully! 🎉")
            else:
                st.warning("Please upload PDF files first")

    
    display_chat_messages()

    
    if st.session_state.pdf_processed:
        
        if prompt := st.chat_input("Ask a question about your PDF"):
            
            st.chat_message("user").markdown(prompt)
            
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            
            response = process_user_question(prompt)
            
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload and process a PDF file in the sidebar to start chatting.")

if __name__ == "__main__":
    main()