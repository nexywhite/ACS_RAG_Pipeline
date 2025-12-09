from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader

import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

import subprocess
import tempfile


# PDF LOADER
def load_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


# TEXT FILE LOADER
def load_text(path):
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


# IMAGE OCR LOADER
def ocr_image(path):
    text = pytesseract.image_to_string(Image.open(path))
    doc = Document(
        page_content=text,
        metadata={"source": os.path.basename(path), "type": "image_ocr"}
    )
    return [doc]


# CHUNKER
def split_documents(documents, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
