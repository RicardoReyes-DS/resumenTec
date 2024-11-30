import asyncio
import pdfplumber
import logging
from io import BytesIO
from typing import Optional
from docx import Document
import magic

async def extract_text(file_content: bytes, file_extension: Optional[str] = None) -> str:
    """
    Extract text from a file, supporting PDF and DOCX formats.

    This function processes the binary content of a file to extract text. 
    It attempts to identify the file type automatically if not provided,
    using the `magic` library.

    Args:
        file_content (bytes): The binary content of the file.
        file_extension (Optional[str]): The file extension ('pdf' or 'docx').

    Returns:
        str: The extracted text from the file.

    Raises:
        ValueError: If the file content is invalid or the file type cannot be determined.
        RuntimeError: If text extraction fails due to unexpected errors.
    """
    if not file_content:
        logging.error("File content is empty or invalid.")
        raise ValueError("The file content provided is empty or invalid.")

    # Auto-detect the file type if no extension is provided
    if not file_extension:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(file_content)
        if mime_type == "application/pdf":
            file_extension = "pdf"
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_extension = "docx"
        else:
            logging.error(f"Unsupported MIME type: {mime_type}")
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

    text = ""

    try:
        if file_extension.lower() == "pdf":
            # Use asyncio.to_thread to handle blocking operations asynchronously
            text = await asyncio.to_thread(_extract_pdf_text, file_content)
        elif file_extension.lower() == "docx":
            text = await asyncio.to_thread(_extract_docx_text, file_content)
        else:
            logging.error(f"Unsupported file extension: {file_extension}.")
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

        logging.info(f"Text extraction completed successfully for {file_extension.upper()} file.")
        return text.strip()

    except Exception as e:
        logging.error(f"Unexpected error during {file_extension.upper()} text extraction: {e}")
        raise RuntimeError(f"Failed to extract text from the {file_extension.upper()} file.") from e


def _extract_pdf_text(file_content: bytes) -> str:
    """Helper function to extract text from PDF files."""
    text = ""
    with pdfplumber.open(BytesIO(file_content)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                logging.info(f"Extracted text from PDF page {page_num}.")
            else:
                logging.warning(f"No text found on PDF page {page_num}.")
    return text


def _extract_docx_text(file_content: bytes) -> str:
    """Helper function to extract text from DOCX files."""
    doc = Document(BytesIO(file_content))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    if paragraphs:
        logging.info(f"Extracted text from DOCX file with {len(paragraphs)} paragraphs.")
        return "\n".join(paragraphs)
    else:
        logging.warning("No text found in the DOCX file.")
        return ""
