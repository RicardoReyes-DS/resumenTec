import logging
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware to allow external requests (e.g., from Postman or browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """Root endpoint for FastAPI"""
    return {"message": "Welcome to the Technical Document Enhancer API"}

@app.post("/process_pdf/")
async def process_pdf_endpoint(file: UploadFile):
    """
    Endpoint to process an uploaded PDF file and improve its text quality.
    Args:
        file (UploadFile): The uploaded PDF file.
    Returns:
        dict: The JSON response containing the processed text or an error message.
    """
    try:
        pdf_content = await file.read()

        # Extract text from PDF
        text = await extract_text(pdf_content)
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to extract text from the PDF file."},
            )

        # Segment extracted text into chunks for individual processing
        chunks = segment_text(text)

        # Improve each text chunk asynchronously
        tasks = [improve_text_gpt(chunk) for chunk in chunks]
        improved_chunks = await asyncio.gather(*tasks)
        improved_text = "\n\n".join(improved_chunks)

        # Further improve the entire text
        improved_full_text = await improve_full_text_gpt(improved_text)

        result = {"improved_text": improved_full_text}
        logging.info("PDF processing completed successfully.")
        return result

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
    try:
        # Read PDF content
        with open(pdf_file.name, "rb") as file:
            pdf_content = file.read()

        # Use asyncio.run to call process_pdf_endpoint with the file content
        response = asyncio.run(process_pdf_endpoint(file=UploadFile(file=pdf_file)))

        # Return error or result
        if "error" in response:
            return response["error"]
        else:
            return response["improved_text"]
    except Exception as e:
        logging.error(f"An error occurred in the Gradio interface: {e}")
        return "An error occurred while processing the document."


# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="Upload your PDF file."),
    outputs=gr.Markdown(label="Processed Output", show_copy_button=True),
    title="Technical Document Enhancer",
    description="Improve technical documents with AI.",
    show_progress="full",
)

# Mount Gradio interface into FastAPI at "/gradio"
from gradio.routes import App as GradioApp

app.mount("/gradio", GradioApp.create_app(iface))

if __name__ == "__main__":
    import uvicorn

    # Run FastAPI (including Gradio) on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=7860)