# utils/pdf_processing

import pdfplumber
import logging
from io import BytesIO
from typing import Optional
from docx import Document
import magic

def extract_text(file_content: bytes, file_extension: Optional[str] = None) -> str:
    """
    Extract text from a file, supporting PDF and DOCX formats.

    This function processes the binary content of a file to extract text.
    It attempts to identify the file type automatically if not provided,
    using 'python-magic' library or by inspecting the content. 

    Args:
        file_content (bytes): The binary content of the file.
        file_extension (Optional[str]): The file extension ('pdf' o 'docx')
                                        If not provided, attempts to detect it
    
    Returns:
        str: The extracted text from the file.

    Logs:
        Logs extraction details for each page (PDF) or paragraph (DOCX).
        Logs error from invalid files or unsupported formats. 

    Raises:
        ValueError: If the file content is invalid of the file type cannot be determined.
        RuntimeError: If text extraction fails to unexpected errors. 
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
            logging.error("Unsuported MIME type: {mime_type}")
            raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

    text = ""

    try:
        if file_extension.lower() == "pdf":
            # Handle PDF file using pdfplumber
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        logging.info(f"Extracted text from PDF page {page_num}.")
                    else:
                        logging.warning(f"No text found on PDF page {page_num}.")
        
        elif file_extension.lower() == "docx":
            # Handle DOCX file using python-docx
            doc = Document(BytesIO(file_content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            if paragraphs:
                text = "\n".join(paragraphs)
