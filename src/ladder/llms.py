from typing import Optional, Literal
from dotenv import load_dotenv
from loguru import logger
import dspy
import sys
import os 

load_dotenv()

# TODO:: test , tracing , history , ..

class BaseLM(dspy.LM):
    """ Base Class for all LLMs

    TODO:: Handle logs , tracing, ...
    """

class OpenAIModel(BaseLM):
    """A wrapper class for dspy.OpenAI."""
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        _check_api(api_key, "OPENAI_API_KEY")
        super().__init__(model=f"openai/{model}", api_key=api_key or os.environ.get("OPENAI_API_KEY"), **kwargs)
  
class GoogleModel(BaseLM):
    """A wrapper class for Google Gemini API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """You can use `genai.list_models()` to get a list of available models."""
        _check_api(api_key, "GOOGLE_API_KEY")
        super().__init__(model=f"gemini/{model}", api_key=api_key or os.environ.get("GOOGLE_API_KEY"), **kwargs)

class ClaudeModel(BaseLM):
    """Copied from dspy/dsp/modules/anthropic.py with the addition of tracking token usage."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=f"anthropic/{model}", api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),  **kwargs)

class DataBricks(BaseLM):
    def __init__(self, model: str, api_key: str,  base_url: Optional[str] = None,  **kwargs):
        _check_api(api_key, "DATABRICKS_API_KEY")
        super().__init__(f"databricks/{model}", api_key=api_key or os.environ.get("DATABRICKS_API_KEY"), base_url=base_url, **kwargs)
       
class LitellmModel(BaseLM):
    def __init__(self, model: str, api_key: str = None,  base_url: Optional[str] = None,  **kwargs):
        """
        Litellm client wrapper for DSPy
        
        Args:
            model: Model name
            base_url: API base URL  "http://localhost:4000")
            api_key: API key 
            **kwargs: Additional completion arguments
        """
        super().__init__(model, api_key=api_key, base_url=base_url, **kwargs)
    
class VLLMModel(LitellmModel):
    """A client compatible with vLLM HTTP server.

    vLLM HTTP server is designed to be compatible with the OpenAI API. Use OpenAI client to interact with the server.
    """

class DeepSeekModel(BaseLM):
    """A wrapper class for DeepSeek API, compatible with dspy.OpenAI and using the OpenAI SDK."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        **kwargs,
    ):
        _check_api(api_key, "DEEPSEEK_API_KEY")
        super().__init__(model=f"deepseek/{model}", api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"), api_base=api_base, **kwargs)

class AzureOpenAIModel(BaseLM):
    """A wrapper class of Azure OpenAI endpoint.
    """
    def __init__(
        self,
        azure_deployment: str,
        api_version: str,
        api_key: str,
        api_base: str,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        _check_api(api_key, "AZURE_OPENAI_API_KEY")
        _check_api(api_version, "AZURE_API_VERSION")
        _check_api(api_base, "AZURE_OPENAI_API_BASE")
        super().__init__(f"azure/{azure_deployment}", 
                         api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
                         api_version=api_version or os.environ.get("AZURE_API_VERSION"),
                         api_base=api_base or os.environ.get("AZURE_OPENAI_API_BASE"),  
                         model_type=model_type,
                         **kwargs)
       
class GroqModel(BaseLM):
    """A wrapper class for Groq API (https://console.groq.com/)"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.groq.com/openai/v1",
        **kwargs,
    ):
        _check_api(api_key, "GROQ_API_KEY")
        super().__init__(model=model, api_key=api_key or os.environ.get("GROQ_API_KEY"), api_base=api_base, **kwargs)
        
class OllamaModel(BaseLM):
    """A wrapper class for dspy.OllamaClient."""

    def __init__(self, 
                 model: str, 
                 api_key: str=None,  
                 base_url: Optional[str] = None,  
                 stream:bool=False,
                 **kwargs):
        """
        OpenAI client wrapper for DSPy
        
        Args:
            model: Model name
            base_url: API base URL  
            api_key: API key 
            **kwargs: Additional completion arguments
        """
        super().__init__(model=f"ollama/{model}", 
                         api_key=api_key, 
                         base_url=base_url,
                         provider="ollama",
                         stream=stream,
                         **kwargs)

class TogetherAIModel(BaseLM):
    """A wrapper class for together ai """

    def __init__(self, 
                 model: str, 
                 api_key: str=None,  
                 base_url: Optional[str] = None,  
                 stream:bool=False,
                 **kwargs):
        """
        Together client wrapper for DSPy
        
        Args:
            model: Model name
            base_url: API base URL  
            api_key: API key 
            **kwargs: Additional completion arguments
        """
        super().__init__(model=f"together_ai/{model}", 
                         api_key=api_key, 
                         base_url=base_url or "https://api.together.xyz/v1",
                         stream=stream,
                         **kwargs)


# TODO:: add replicate llm 

def _check_api(api_key:str, api_env_name:str):
    if not api_key and not os.environ.get(api_env_name):
        logger.error(f"{api_env_name} must be provided either as an argument or as an environment variable {api_env_name}")
        # raise ValueError(
        #     f"{api_env_name} must be provided either as an argument or as an environment variable {api_env_name}"
        # )
        sys.exit(1)
