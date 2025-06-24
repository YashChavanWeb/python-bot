from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from PIL import Image
import pytesseract
import io
import re
import json
from datetime import datetime

# import speech_recognition as sr # Commented out due to deployment challenges with audio recording
from pydantic import BaseModel

# ==== Gemini Config ====
# It's highly recommended to use environment variables for API keys in production
API_KEY = os.getenv(
    "GEMINI_API_KEY", "AIzaSyA58up6mb0EppG3dI0lT2WYct4Om9aEQKw"
)  # Use os.getenv for API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="Your responses should be concise, short, direct, and educational, answering only educational queries with a 120 to 200 word limit.",
)

# ==== FastAPI App ====
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
    "https://python-bot-bw2k.onrender.com",  # Add your Render frontend URL if it's different or for direct access
    # You might want to remove "*" in production for security
    "*",  # Keep for development ease, but be mindful for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
chat_history = []
document_context = ""

# Directories (might not be needed if not saving audio files on server)
# AUDIO_DIR = "audio_files"
# OUTPUT_DIR = "outputs"
# AUDIO_FILENAME = "audio.wav"
# JSON_FILENAME = "transcription.json"

# os.makedirs(AUDIO_DIR, exist_ok=True) # Commented out
# os.makedirs(OUTPUT_DIR, exist_ok=True) # Commented out

# audio_path = os.path.join(AUDIO_DIR, AUDIO_FILENAME) # Commented out
# json_path = os.path.join(OUTPUT_DIR, JSON_FILENAME) # Commented out


# Pydantic model for response (commented out if audio recording is removed)
# class TranscriptionResponse(BaseModel):
#     timestamp: str
#     audio_file: str
#     transcription: str


# ==== Document Handlers ====
def extract_text_from_pdf(content):
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_txt(content):
    return content.decode("utf-8")


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_pptx(file):
    prs = Presentation(file)
    return "\n".join(
        shape.text
        for slide in prs.slides
        for shape in slide.shapes
        if hasattr(shape, "text")
    )


def extract_text_from_image(file):
    img = Image.open(file)
    return pytesseract.image_to_string(img)


def get_text_from_file(uploaded_file: UploadFile):
    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    content = uploaded_file.file.read()
    uploaded_file.file.seek(0)

    if ext == ".pdf":
        return extract_text_from_pdf(content)
    elif ext == ".txt":
        return extract_text_from_txt(content)
    elif ext == ".docx":
        return extract_text_from_docx(uploaded_file.file)
    elif ext == ".pptx":
        return extract_text_from_pptx(uploaded_file.file)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(uploaded_file.file)
    else:
        return "Unsupported document type"


# ==== Gemini Logic ====
def generate_response(user_message, history):
    global document_context
    full_query = user_message
    if document_context:
        full_query = f"Based on the following document content:\n\n{document_context}\n\nAnd the user's query: {user_message}"

    try:
        # Gemini API expects chat history in specific format
        # Your current history is (human, ai) tuples. Let's convert it.
        formatted_history = []
        for human_msg, ai_msg in history:
            formatted_history.append({"role": "user", "parts": [human_msg]})
            formatted_history.append({"role": "model", "parts": [ai_msg]})

        # Start a chat session with the model and provide the history
        chat_session = model.start_chat(history=formatted_history)
        response = chat_session.send_message(full_query)
        bot_response = response.text.strip()

        return format_bot_response(bot_response)

    except Exception as e:
        print(f"Error during Gemini API call: {e}")  # Log the actual error
        return f"An error occurred with the AI model: {e}. Please try again or rephrase your query."


def format_bot_response(response: str) -> str:
    cleaned_response = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
    cleaned_response = re.sub(r"\*(.*?)\*", r"\1", cleaned_response)
    cleaned_response = cleaned_response.replace("<br>", "\n")
    cleaned_response = re.sub(r"<[^>]+>", "", cleaned_response)
    cleaned_response = cleaned_response.replace("\\&", "&").replace("&nbsp;", " ")
    cleaned_response = re.sub(r"(?<=\d\.)\s*", " ", cleaned_response)
    cleaned_response = re.sub(r"\n(?=\d\.)", "\n\n", cleaned_response)
    cleaned_response = re.sub(r"\n(?=- )", "\n\n", cleaned_response)
    cleaned_response = re.sub(r"\n{2,}", "\n\n", cleaned_response).strip()
    return cleaned_response


# ==== Endpoints ====


@app.get("/")
async def read_root():
    return {"message": "QuizzyBot API is running!"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_context
    try:
        document_context = get_text_from_file(file)
        if document_context == "Unsupported document type":
            raise HTTPException(status_code=400, detail="Unsupported document type")
        return {
            "message": "Document uploaded successfully.",
            "preview": document_context[:300],
        }
    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@app.post("/chat")
async def chat(message: str = Form(...)):
    global document_context, chat_history
    query = (
        f"Based on the document:\n{document_context}\nUser asked: {message}"
        if document_context
        else message
    )

    try:
        response = generate_response(query, chat_history)
        if not response:
            raise HTTPException(status_code=500, detail="Empty response from Gemini.")

        chat_history.append((message, response))
        return {"response": response}
    except Exception as e:
        print(f"Gemini API error in /chat: {str(e)}")  # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


@app.post("/clear")
async def clear():
    global document_context, chat_history
    chat_history = []
    document_context = ""
    return {"message": "Context and chat history cleared."}


# @app.post("/record_audio", response_model=TranscriptionResponse)
# async def record_audio():
#     # This endpoint is commented out because sounddevice and speech_recognition
#     # often cause issues in serverless or containerized environments like Render.
#     # If you need audio transcription, consider recording audio in the frontend
#     # and sending it as a file to a new backend endpoint for transcription.
#     return HTTPException(status_code=501, detail="Audio recording is not supported on this server.")

# To run the app with Uvicorn, you need to use a __main__ block
# This is crucial for Render to pick up your application
if __name__ == "__main__":
    import uvicorn

    # Get the port from the environment variable, default to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", host="0.0.0.0", port=port, reload=True
    )  # Assuming your file is named main.py
