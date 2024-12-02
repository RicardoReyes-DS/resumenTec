# main.py

import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import gradio as gr
import asyncio
from utils.logging_config import setup_logging
from utils.pdf_processing import extract_text
from utils.text_segmentation import segment_text
from utils.openai_api import improve_text_gpt, improve_full_text_gpt

# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Technical Document Enhancer API"}

@app.post("/process_pdf/")
async def process_pdf_endpoint(
    pdf_content: bytes,
) -> dict:
    """
    Endpoint to process an uploaded PDF file and improve its text quality.

    Args:
        pdf_content (bytes): The PDF file content as bytes.

    Returns:
        dict: The JSON response containing the processed text or an error message.
    """

    try:
        # Extract text from PDF
        text = await extract_text(pdf_content)

        if not text:
            return {"error": "Failed to extract text from the PDF file."}
        
        # Segment extracted text into chunks for individual processing
        chunks = segment_text(text)

        # Improve each text chunk asynchronously
        tasks = [improve_text_gpt(chunk) for chunk in chunks]
        improved_chunks = await asyncio.gather(*tasks)
        improved_text = '\n\n'.join(improved_chunks)

        # Further improve the entire text using the new function
        improved_full_text = await improve_full_text_gpt(improved_text)

        result = f"**Improved text:**\n\n{improved_full_text}"

        logging.info("PDF processing completed successfully.")
        return {"result": result}
    
    except Exception as e:
        logging.error(f"An error occurred during PDF processing: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing.")
    
def gradio_interface(pdf_file):
    """
    Gradio interface to handle PDF processing and output the enhanced text.

    Args:
        pdf_file: The uploaded PDF file from Gradio
    
    Returns:
        str: Processed and improved text.
    """
    # Check if 'pdf_file' is a path and read the content
    with open(pdf_file.name, "rb") as file:
        pdf_content = file.read()

    # Use asyncio.run to call process_pdf_endpoint with the file content
    response = asyncio.run(process_pdf_endpoint(pdf_content=pdf_content))

    # Return error or result
    if 'error' in response:
        return response['error']
    else:
        return response['result']
    
    
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="Upload your PDF"),
    outputs=gr.Markdown(label="Processed Output", show_copy_button=True),
    title="Technical Document Enhancer",
    description="Improve technical documents with AI."
)

iface.launch()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
