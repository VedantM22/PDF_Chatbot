import streamlit as st
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.docstore.document import Document  # Import Document object
from dotenv import load_dotenv
load_dotenv()

# Define the LLM model with Groq
groq = os.getenv('GROQ_API_KEY')
if not groq:
    st.error("GROQ_API_KEY not set. Please check your environment variables.")
    st.stop()

llm = ChatGroq(groq_api_key=groq, model="Gemma-7b-it")

# Define the embedding model to convert the PDF (text) into vectors
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the chat prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions of the user as you are a helpful assistant.
    Answer the questions as accurately as possible.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Create a function to generate embeddings from the uploaded PDF
def create_vectors(model, pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Convert the extracted text into a Document object, required by Langchain
    docs = [Document(page_content=text)]
    
    # Split the documents for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)
    
    # Create vectors using FAISS
    vectors = FAISS.from_documents(final_docs, model)
    return vectors

# Streamlit app
st.title('Let’s have a conversation with your PDF')

# Uploading the PDF
pdf = st.sidebar.file_uploader("Choose a PDF file:", type="pdf")
if pdf is not None:
    st.write('PDF uploaded successfully')
    vectors = create_vectors(model=embed, pdf_file=pdf)

    # Allow the user to input a query
    query = st.text_input('Enter your query and press "Enter" to generate a response.')

    if query:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        response = retriever_chain.invoke({'input': query})

        st.write(response['answer'])
else:
    st.write('Please upload a PDF file.')
