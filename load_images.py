import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from fpdf import FPDF
from logger_utils import setup_logger
logger = setup_logger()

# Set the Tesseract-OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_images_to_pdf(input_path):
    logger.info(f"Processing file: {input_path}")
    output_pdf_path = input_path.replace('.pdf', '_text.pdf') if input_path.endswith('.pdf') else input_path + '.pdf'
    images = []

    # If input is a PDF, convert pages to images
    if input_path.endswith('.pdf'):
        images = convert_from_path(input_path, poppler_path=r"lib\poppler-24.08.0\Library\bin")
    else:  # If input is an image file
        images.append(Image.open(input_path))

    # Extract text and save as PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for image in images:
        text = pytesseract.image_to_string(image)

        # Safely handle unsupported characters by encoding with latin1
        safe_text = text.encode("latin1", errors="replace").decode("latin1")

        pdf.add_page()
        pdf.multi_cell(0, 10, safe_text)

    pdf.output(output_pdf_path)
    logger.info(f"Processed and saved text-based PDF: {output_pdf_path}")
