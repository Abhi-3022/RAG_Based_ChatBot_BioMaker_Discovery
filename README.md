# 🧬 RAG-Based Biomarker Discovery Multi-Document Assistant

The **Biomarker Discovery Multi-Document Assistant** is an AI-powered tool that enables researchers to extract meaningful insights from multiple biomedical research documents. It uses advanced AI models to provide comprehensive answers to questions based on the content of the documents.

---

## 🌟 Features

- **Multi-PDF Processing**: Automatically processes multiple PDFs stored in a specified directory.
- **Biomedical Research Insights**: AI-powered question answering for biomarker discovery and analysis.
- **Efficient Text Processing**: Splits large research documents into manageable chunks for analysis.
- **Interactive Chat Interface**: Allows users to query documents interactively.
- **Comprehensive Metadata**: Displays file metadata (e.g., name, size, number of documents).

---

## 🛠️ Project Overview

This project assists researchers in extracting biomarkers or other critical insights from large collections of biomedical research documents. It leverages **LangChain** for text processing, **FAISS** for similarity searches, and **Google Generative AI** for embeddings and conversational capabilities.

---

## 📥 Sample Input

**Example Question**:
> "What are the key biomarkers identified in these studies for cancer treatment?"

**PDF Directory**:
- `Study1.pdf` (1.2 MB)
- `Study2.pdf` (3.4 MB)

---

## 📤 Sample Output

**User Input**:
> "What biomarkers are associated with Alzheimer’s disease in the documents?"

**Assistant Output**:
> "The studies indicate that beta-amyloid plaques and tau proteins are key biomarkers associated with Alzheimer's disease. Additional markers include inflammatory cytokines and oxidative stress indicators."

---

## 🚀 How to Run the Project

### Prerequisites
1. Python 3.8 or later.
2. A valid **Google API Key** with access to Google Generative AI.
3. A folder containing PDF files for processing (default: `BioMark/`).

### Steps to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/biomarker-assistant.git
   cd biomarker-assistant
   
   pip install -r requirements.txt
   
   streamlit run app.py

## ⚙️ How It Works
### PDF Metadata Extraction:

- Extracts file names, sizes, and counts from the BioMark/ directory.
### Text Processing:

- Reads and combines text from all PDFs.
- Splits the text into chunks using LangChain’s RecursiveCharacterTextSplitter.
### Vector Store Creation:

- Generates embeddings using Google Generative AI.
- Creates and stores a FAISS index for similarity searches.
### Question Answering:

- Retrieves the most relevant text chunks for a given question.
- Uses a custom prompt to generate detailed responses.
## 👀 Example Scenarios
- **Biomedical Research:** Query large collections of studies for biomarkers or treatment details.
- **Healthcare Analysis:** Extract insights from clinical trial reports.
- **Academic Use:** Quickly summarize or extract key information from published papers.
