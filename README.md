# PDF Chatbot with Streamlit and Langchain

## Overview
The PDF Chatbot is an innovative web application designed to facilitate interactive conversations with the content of PDF documents. Built using Streamlit and Langchain, this project allows users to seamlessly upload a PDF file and engage in a natural language dialogue about its content.

The chatbot uses advanced language model processing with the Groq API and incorporates embedding techniques from Hugging Face to convert the document's text into vector representations. By leveraging these technologies, the chatbot can efficiently retrieve relevant information and provide accurate responses to user queries.

## Features
PDF Upload: Users can upload PDF files containing textual information, such as research papers, reports, or manuals.
Natural Language Querying: Users can ask questions related to the content of the PDF, and the chatbot will respond contextually.
Contextual Responses: The chatbot provides answers based on the extracted text, ensuring relevance and accuracy.
User-Friendly Interface: The web application is built with Streamlit, offering a clean and intuitive user interface for easy interaction.
Architecture
The architecture of the PDF Chatbot consists of the following components:

##PDF Processing:

PDF Extraction: Uses the PyPDF2 library to read and extract text from the uploaded PDF file.
Text Splitting: Utilizes the RecursiveCharacterTextSplitter to break down the extracted text into manageable chunks for efficient embedding and retrieval.
Embedding Model:

Integrates the Hugging Face model sentence-transformers/all-MiniLM-L6-v2 for generating embeddings that convert the textual data into vector representations, facilitating semantic understanding.
Retrieval Chain:

Implements a retrieval mechanism using the FAISS library to store and query the embeddings, enabling quick access to relevant document sections based on user queries.
Language Model:

Leverages the Groq API with the ChatGroq model to generate contextual responses based on the input query and retrieved document content.
Streamlit Interface:

Provides a web-based interface for users to upload PDF files, enter queries, and view responses.
