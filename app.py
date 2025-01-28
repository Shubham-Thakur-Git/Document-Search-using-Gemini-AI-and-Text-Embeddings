import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from docx import Document  # Import python-docx for handling .docx files

# Page configuration
st.set_page_config(page_title="Document Search AI", page_icon="📚", layout="wide")


# Custom CSS for styling
st.markdown("""
    <style>
        /* Apply dark gray background to entire app */
        .stApp {
            background-color: #2E2E2E;  /* Dark Gray */
        }
        
        /* Override background of the main content area */
        div[data-testid="stAppViewContainer"] {
            background-color: #2E2E2E;
        }

        /* Set background for sidebar */
        div[data-testid="stSidebar"] {
            background-color: #2E2E2E;
        }

        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
        .stDataFrame {
            border-radius: 15px;
            overflow: hidden;
        }
        .title-text {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .content-box {
            background-color: #1E1E1E;  /* Slightly darker gray for contrast */
            color: white;
            padding: 15px;
            border-radius: 10px;
            white-space: pre-wrap;
        }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown("<div class='title-text'>Document Search using Gemini AI</div>", unsafe_allow_html=True)

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCOFymdoboKLN7vdoGoVr_97LxBPtMGkNs")

# Document titles based on filenames
document_titles = {
    "Biology.docx": "Human Digestive System",
    "Physics.docx": "Fundamentals of Classical Mechanics",
    "Chemistry.docx": "Introduction to Organic Chemistry"
}

# Function to extract text while maintaining the structure of the document
def extract_text_with_formatting(doc):
    formatted_text = ""
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            formatted_text += f"### {para.text}\n\n"  # Add Markdown formatting for headings
        elif para.text.strip().startswith('-') or para.text.strip().endswith(':'):
            formatted_text += f"- {para.text}\n"  # Handle bullet points and key terms
        else:
            formatted_text += f"{para.text}\n\n"  # Regular paragraph formatting
    return formatted_text

# File uploader
uploaded_files = st.file_uploader("Upload your documents", type=["docx"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        doc_title = document_titles[uploaded_file.name]
        doc = Document(uploaded_file)
        formatted_content = extract_text_with_formatting(doc)
        
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=formatted_content,
            task_type="retrieval_document",
            title=doc_title
        )['embedding']

        documents.append({"Title": doc_title, "Text": formatted_content, "Embeddings": embedding})

    df = pd.DataFrame(documents)
    with st.expander("View Uploaded Documents"):
        st.dataframe(df[['Title', 'Text']])


# Query input with placeholder
query = st.text_input("Write your query", "", placeholder="Enter your query here...")

if st.button("🔍 Find most relevant document according to the query"):
    if query and 'df' in locals():
        query_vector = genai.embed_content(model="models/embedding-001", content=query, task_type="retrieval_query")['embedding']

        def similarity(document_vector, query_vector):
            return np.dot(document_vector, query_vector)

        similarities = [similarity(i, query_vector) for i in df['Embeddings']]
        max_index = np.argmax(similarities)

        st.subheader("Most Relevant Document:")
        st.markdown(f"**Document Name:** {[k for k, v in document_titles.items() if v == df.iloc[max_index]['Title']][0]}")
        st.markdown(f"**Title:** {df.iloc[max_index]['Title']}")
        st.markdown(f"<div class='content-box'>{df.iloc[max_index]['Text']}</div>", unsafe_allow_html=True)
