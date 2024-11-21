# utils/openai_api.py

import openai
import logging
from config import settings

# Set OpenAI API key from configuration settings
openai.api_key = settings.OPENAI_API_KEY

async def improve_text_gpt(
        chunk: str, 
        model: str = "gpt-4o-mini", 
        temperature: float = 0.7,
        max_tokens: int = 8000
) -> str:
    """
    Improve a chunk of text using OpenAI's GPT model.

    Args:
        chunk (str): The block of text to be improved.
        model (str): The model to be used for text improvement, 
                     defaults to 'gpt-4o-mini'.
        temperature (float) : The temperature parameter to control creativity,
                              defaults to 0.7.
        max_tokens (int): The maximun number of tokens in the response, 
                          defaults to 8000
    """
    try:
        logging.info(f"Sending requests to OpenAI API to improve the text with model: {model}")

        # Prepare the messages for the OpenAI GPT API call
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un experto en redacción de textos técnicos en ingeniería.\n"
                    "Tu tarea es mejorar el siguiente texto para audiencias de ejecutivos, y líderes técnicos.\n"
                    "Mantén un tono profesional, técnico, preciso y persuasivo.\n"
                    "No agregues comentarios, análisis, observaciones ni conclusiones personales.\n"
                    "Devuelve únicamente el texto mejorado, estrucutrado de manera clara y coherente."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Por favor mejora el siguiente grupo de oraciones.\n"
                    "Cada grupo está previamente segmentado y agrupado por temática mediante un"
                    "algoritmo de aprendizaje automático, como KMeans.\n"
                    "El texto generado está dirigido a ejecutivos y líderes técnicos.\n"
                    "Considera los siguientes puntos:\n"
                    "1. Cada grupo debe ser tratado como un conjunto coherente,"
                    "asegurando que el mensaje de cada sección se mantenga alineado con la temática que representa.\n"
                    "2. No alteres, remplaces, ni reformules nombres propios o cifras técnicas."
                    "Manten esa información tal cual.\n"
                    "3. Asegurate que el tono sea apropiado para ejecutivos y líderes técnicos."
                    "utilizando un lenguage conciso y alineado con loas mejores prácticas de comunicación empresarial.\n"
                    "4. Presta especial atención a la concordancia gramatical, la estructura de las frases"
                    "y la eliminación de de ambigüedades.\n"
                    "5. Asegúrate que cada grupo de oraciones fluya de manera natural y coherente,"
                    "revisando que las ideas se desarrollen de manera progresiva y sin saltos lógicos.\n"
                    "6. Confirma que todos los términos técnicos sean utilizados correctamente,"
                    "alineándose con el contexto ingenieril del texto.\n"
                    "7. Identifica frases repetitivas y reformúlalas para mantener concisión, sin sacrificar"
                    "información relevante o el impacto persuasivo del texto.\n"
                    "8. Asegúrate de que el texto esté formateado para una presentación profesional,"
                    "con uso adecuado de títulos, subtítulos, viñetas, y una disposición clara su fácil lectura.\n"
                    f"A continuación, se presenta el grupo de oraciones: \n\n {chunk}"
                )
            }
        ]

        # Call the OpenAI API to improve the text
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
        )

        improved_text = response.choices[0].message.content.strip()
        logging.info("Text improvement completed succesfully.")
        return improved_text
    
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return chunk # Return original chunk on API error
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return chunk # Return original chunk on unexpected error