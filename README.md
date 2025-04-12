# Document and Code Plagiarism Detector

## Description
This application is designed to detect plagiarism and similarities between documents and code. It supports various document formats (TXT, PDF, DOCX, CSV) and programming languages (C, C++, Python, JavaScript). Using advanced NLP models and code analysis techniques, this tool compares the text and code sections of two files to provide a detailed similarity report.

## Features
- **Text Plagiarism Detection**: Compares two documents and provides an overall similarity score based on text content.
- **Code Similarity Detection**: Compares code files (in C, C++, Python, and JavaScript) and generates a similarity score based on code structure and functionality.
- **File Conversion**: Supports code conversion (e.g., from Python to C++) using the Gemini API.
- **Plagiarism Report**: Generates and allows downloading a detailed markdown report on document and code similarities.

## Technologies Used
- **Streamlit**: Framework for building the web interface.
- **PyPDF2**: Library for extracting text from PDF files.
- **python-docx**: Library for extracting text from DOCX files.
- **Pandas**: Data analysis library, used for processing CSV files.
- **Transformers**: Hugging Face's transformer models for BERT-based text similarity computation.
- **Gemini API**: Used for code conversion between programming languages.
- **Torch**: For working with BERT models and computing similarity.

## Installation

To run this application, make sure you have Python 3.8 or higher installed. You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt


