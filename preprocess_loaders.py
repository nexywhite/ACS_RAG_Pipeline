from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import pytesseract
from PIL import Image
import os
import subprocess
import tempfile

# Configure paths for Tesseract and FFmpeg
# (mainly for me (Sofia), since I have troubles with PATH on Windows sometimes)
# Adjust these paths as necessary for your environment
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# FFMPEG_PATH = "C:\\Users\\skons\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0.1-full_build\\bin\\ffmpeg.exe"

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


# AUDIO/VIDEO TRANSCRIPTION LOADER
def transcribe_media(path, whisper_model="tiny"):
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav = tmp_file.name
    
    # Convert media to WAV using ffmpeg
    cmd = [
        "ffmpeg",           # Change to FFMPEG_PATH if necessary
        "-y",               # Overwrite output files without asking
        "-i", path,         # Input file
        "-vn",              # Ignore video stream
        "-ar", "16000",     # Resample to 16kHz
        "-ac", "1",         # Force mono channel
        "-f", "wav",        # Output format .wav
        tmp_wav,            # Output file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribe using Whisper
    import whisper
    model = whisper.load_model(whisper_model)
    result = model.transcribe(tmp_wav)
    text = result.get("text", "").strip()

    # Clean up temporary WAV file
    try:
        os.remove(tmp_wav)
    except OSError:
        pass

    return [
        Document(
            page_content=text,
            metadata={"source": os.path.basename(path), "type": "media_transcription"}
        )
    ]


# CHUNKER
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
