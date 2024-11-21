# config.py

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Manages environment variables for configuration. 

    Attributes:
        OPENAI_API_KEY (str): The API key for accesing OpenAI's services,
                              fetched from "OPENAI_API_KEY" environment varible  
    """
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY", description="API key for OpenAI services")

    class Config:
        """
        Configuration for the Settings class.

        This configuration class can be used to specify how environment variables are loaded, 
        in this case directly from environment '.env'
        """
        env_file = ".env" # Allows settings to be loaded from a .env file if available

# Create a settings instance
settings = Settings()