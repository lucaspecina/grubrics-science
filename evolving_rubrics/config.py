"""
Configuration module for DR-Tulu Evolving Rubrics.

Handles environment variables, API configuration, and client initialization.
"""

import os

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import OpenAI clients
try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    raise ImportError("openai package is required. Install with: pip install openai")


# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

USE_AZURE = os.environ.get("USE_AZURE_OPENAI", "false").lower() == "true"
AZURE_API_BASE = os.environ.get("AZURE_API_BASE", "https://development-cursor-models.openai.azure.com/")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-02-15-preview")

RUBRIC_GENERATION_MODEL = os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4o-mini")
RUBRIC_JUDGE_MODEL = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def get_client():
    """
    Initialize and return the appropriate OpenAI client.
    
    Returns:
        AsyncOpenAI or AsyncAzureOpenAI client instance
    
    Raises:
        ValueError: If Azure is enabled but API key is not configured
    """
    if USE_AZURE:
        if not AZURE_API_KEY:
            raise ValueError("USE_AZURE_OPENAI=true but AZURE_API_KEY is not configured")
        return AsyncAzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_API_BASE.rstrip('/'),
            api_key=AZURE_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        )
    else:
        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

