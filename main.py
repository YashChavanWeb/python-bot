from fastapi import FastAPI, UploadFile, File, Form
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

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="Your responses should be concise, short, direct, and educational, answering only educational queries with a 120 to 200 word limit.",
)

# ==== FastAPI App ====
app = FastAPI()

# Enable CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins, adjust in production for specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Session state (for a single instance, not truly scalable for multiple users without a database)
chat_history = []
document_context = ""


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
    # pytesseract needs Tesseract OCR engine installed on the system.
    # For Render, you might need to use a custom Dockerfile or ensure Tesseract is available.
    # A common approach is to use a pre-built Docker image that includes Tesseract.
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


# ==== Quizzy Bot Logic ====


def generate_response(user_message, history):
    global document_context
    full_query = user_message
    if document_context:
        full_query = f"Based on the following document content:\n\n{document_context}\n\nAnd the user's query: {user_message}"

    try:
        prompt_with_history = ""
        for human, ai in history:
            prompt_with_history += f"Human: {human}\nAI: {ai}\n"
        prompt_with_history += f"Human: {full_query}\nAI:"

        # Sending the prompt to Gemini for processing
        response = model.generate_content(prompt_with_history)
        bot_response = response.text.strip()

        # Format the bot response (add some basic HTML structure for clarity)
        structured_response = format_bot_response(bot_response)

        history.append((user_message, structured_response))
        return structured_response

    except Exception as e:
        return f"An error occurred: {e}. Please try again or rephrase your query."


def format_bot_response(response: str) -> str:
    """
    Formats the Gemini response by:
    - Removing markdown and HTML
    - Ensuring clear line breaks between bullet/numbered points
    - Fixing special characters like \&
    - Maintaining clean, readable spacing
    """
    # Remove markdown formatting
    cleaned_response = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
    cleaned_response = re.sub(r"\*(.*?)\*", r"\1", cleaned_response)

    # Remove HTML tags and replace <br> with line breaks
    cleaned_response = cleaned_response.replace("<br>", "\n")
    cleaned_response = re.sub(r"<[^>]+>", "", cleaned_response)

    # Fix encoded characters
    cleaned_response = cleaned_response.replace("\\&", "&").replace("&nbsp;", " ")

    # Add line breaks between numbered or bullet items if not already present
    cleaned_response = re.sub(r"(?<=\d\.)\s*", " ", cleaned_response)  # e.g. 1. Text
    cleaned_response = re.sub(
        r"\n(?=\d\.)", "\n\n", cleaned_response
    )  # Ensure spacing before numbers
    cleaned_response = re.sub(
        r"\n(?=- )", "\n\n", cleaned_response
    )  # Ensure spacing before bullets

    # Normalize multiple newlines
    cleaned_response = re.sub(r"\n{2,}", "\n\n", cleaned_response).strip()

    return cleaned_response


# ==== API Endpoints ====


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_context
    try:
        document_context = get_text_from_file(file)
        # Provide a preview of the uploaded document (first 300 characters)
        return {
            "message": "Document uploaded successfully.",
            "preview": document_context[:300],
        }
    except Exception as e:
        return {"error": f"Error uploading document: {str(e)}"}


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
            return {"error": "Empty response from Gemini."}

        chat_history.append((message, response))
        return {"response": response}
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}


@app.post("/clear")
async def clear():
    global document_context, chat_history
    chat_history = []
    document_context = ""
    return {"message": "Context and chat history cleared."}
