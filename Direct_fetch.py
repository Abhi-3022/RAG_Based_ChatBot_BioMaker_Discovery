import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import uuid

GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'  
genai.configure(api_key=GOOGLE_API_KEY)

PDF_DIRECTORY = 'BioMark/'

def get_pdf_metadata(pdf_directory):
    """Extract metadata such as file names, sizes, and number of files."""
    metadata = {
        "file_count": 0,
        "files": []
    }

    pdf_paths = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in directory: {pdf_directory}")

    metadata["file_count"] = len(pdf_paths)

    for pdf_path in pdf_paths:
        file_info = {
            "name": os.path.basename(pdf_path),
            "size": os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
        }
        metadata["files"].append(file_info)

    return metadata

def get_pdf_texts(pdf_directory):
    """Extract text content from all PDFs in the directory."""
    text = ""
    pdf_paths = glob.glob(os.path.join(pdf_directory, '*.pdf'))

    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a conversational chain for biomarker research queries."""
    prompt_template = """
    Provide a comprehensive and precise answer to the biomedical research question based on the given context.
    If the answer is not available in the provided context, clearly state "Information not found in the current research documents."

    Context:\n {context}\n
    Question: \n{question}\n
    Detailed Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_question(user_question):
    """Process user question and retrieve answer from PDF contexts."""
    if not os.path.exists("faiss_index"):
        st.error("Please ensure the PDFs have been processed!")
        return "Error: PDFs not processed"

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response['output_text']
    except Exception as e:
        return f"An error occurred during biomarker analysis: {e}"

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
    """Main Streamlit application for Biomarker Discovery."""
    st.set_page_config(
        page_title="Biomarker Discovery Multi-Document Assistant",
        page_icon="🧬",
        layout="centered"
    )

    initialize_session_state()

    st.title("🧬 Biomarker Discovery Research Assistant")
    st.caption("Analyze multiple medical research documents and extract comprehensive biomarker insights!")

    try:
        with st.spinner("Initializing Biomarker Research Documents..."):
            metadata = get_pdf_metadata(PDF_DIRECTORY)
            raw_text = get_pdf_texts(PDF_DIRECTORY)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state.pdf_processed = True

            st.success("Research Documents Processed Successfully! 🔬")
            st.write(f"Number of PDFs processed: {metadata['file_count']}")
            for file in metadata['files']:
                st.write(f"- {file['name']} ({file['size']:.2f} MB)")
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        st.session_state.pdf_processed = False

    display_chat_messages()

    if st.session_state.pdf_processed:
        if prompt := st.chat_input("Ask a question about the biomarker research"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = process_user_question(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Unable to initialize the biomarker research documents. Please check the directory path.")

if __name__ == "__main__":
    main()
