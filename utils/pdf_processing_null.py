#utils/pdf_processing_null.py - Esta opción se descarta por ahora

import pdfplumber
import logging
import os
from tempfile import NamedTemporaryFile

async def extract_text(pdf_file_content: bytes) -> str:
    """
    Asynchronously extracts text from a PDF file

    This function reads a PDF file asynchronously from disk and processes its contents 
    without blocking the main event loop. Text extraction is performed in a separate thread
    to handle the synchronous 'pdfplumber' library.

    Args:
        pdf_file_path (str): Path to the PDF file
    
    Returns:
        str: A string containing the extracted text from the PDF, with each page's text
        separated by two newline characters.
    
    Logs:
        Logs an informational message when text extraction completes succesfully.
        Logs an error message with details if the text extraction fails

    Raises:
        FileNotFoundError: If the specified PDF file cannot be found 
        pdfplumber.exceptions.PDFSyntaxError: If the PDF file is corrupt or has syntax issues
    """
    text = ""
    try:
        # Write pdf_file_content to a temporary file for pdfplumber to read
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file_content)
            temp_pdf_path = temp_pdf.name
        
        # Open the PDF file using pdfplumber to extract text
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text:
                    logging.warning(f"No text found on page {page_num}.")
                text += page_text + "\n\n"

        logging.info("Text extraction completed succesfully")
    
    except pdfplumber.PDFSyntaxError as syntax_error:
        logging.error(f"Failed to parse PDF due to syntax error: {syntax_error}")
        return "Failed to parse PDF: Syntax error"

    except Exception as e:
        logging.error(f"Unexpected error during text extraction: {e}")
        return "Failed to extract text from the PDF file."
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
    
    return text