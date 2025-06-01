from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel
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
        super().__init__(model=model, api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"), api_base=api_base, **kwargs)

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

    def __init__(self, model: str, api_key: str=None,  base_url: Optional[str] = None,  **kwargs):
        """
        OpenAI client wrapper for DSPy
        
        Args:
            model: Model name
            base_url: API base URL  
            api_key: API key 
            **kwargs: Additional completion arguments
        """
        super().__init__(model, api_key=api_key, base_url=base_url, **kwargs)

class HFModel(BaseLM):
    """
    Wrapper for Hugging Face models, matching the BaseLM interface.
    """

    # TODO:: test this class 

    def __init__(
        self,
        model: str,
        checkpoint: Optional[str] = None,
        is_client: bool = False,
        hf_device_map: Literal["auto", "balanced", "balanced_low_0", "sequential"] = "auto",
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.provider = "hf"
        self.is_client = is_client
        self.device_map = hf_device_map
        self.checkpoint = checkpoint

        if not self.is_client:
            try:
                import torch
                from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "You need to install Hugging Face transformers library to use HF models.",
                ) from exc
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                architecture = AutoConfig.from_pretrained(model).__dict__["architectures"][0]
                self.encoder_decoder_model = ("ConditionalGeneration" in architecture) or ("T5WithLMHeadModel" in architecture)
                self.decoder_only_model = ("CausalLM" in architecture) or ("GPT2LMHeadModel" in architecture)
                assert self.encoder_decoder_model or self.decoder_only_model, f"Unknown HuggingFace model class: {model}"
                self.tokenizer = AutoTokenizer.from_pretrained(model if checkpoint is None else checkpoint)

                AutoModelClass = AutoModelForSeq2SeqLM if self.encoder_decoder_model else AutoModelForCausalLM
                if checkpoint:
                    if self.device_map:
                        self.model_instance = AutoModelClass.from_pretrained(checkpoint, device_map=self.device_map)
                    else:
                        self.model_instance = AutoModelClass.from_pretrained(checkpoint).to(self.device)
                else:
                    if self.device_map:
                        self.model_instance = AutoModelClass.from_pretrained(model, device_map=self.device_map)
                    else:
                        self.model_instance = AutoModelClass.from_pretrained(model).to(self.device)
                self.drop_prompt_from_output = False
            except Exception:
                self.model_instance = AutoModelForCausalLM.from_pretrained(
                    model if checkpoint is None else checkpoint,
                    device_map=self.device_map,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model)
                self.drop_prompt_from_output = True

    def _generate(self, prompt, **kwargs):
        assert not self.is_client
        import torch

        def _openai_to_hf(**kwargs):
            mapping = {
                "max_tokens": "max_new_tokens",
                "temperature": "temperature",
                "do_sample": "do_sample",
                "top_p": "top_p",
                "num_return_sequences": "num_return_sequences",
                "n": "num_return_sequences",
            }
            hf_kwargs = {}
            for k, v in kwargs.items():
                if k in mapping:
                    hf_kwargs[mapping[k]] = v
            return hf_kwargs

        kwargs = {**_openai_to_hf(**self.kwargs), **_openai_to_hf(**kwargs)}

        if isinstance(prompt, dict):
            try:
                prompt = prompt['messages'][0]['content']
            except (KeyError, IndexError, TypeError):
                raise ValueError("Failed to extract 'content' from the prompt.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model_instance.generate(**inputs, **kwargs)
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [
            {"text": c}
            for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        # Mimic OpenAI response format
        class DummyResponse:
            def __init__(self, prompt, completions):
                self.choices = [type("Choice", (), c) for c in completions]
                self.usage = {"prompt_tokens": inputs.input_ids.numel(), "completion_tokens": outputs.numel()}
                self.model = self.model_instance.__class__.__name__
        return DummyResponse(prompt, completions)

    def forward(self, prompt=None, messages=None, **kwargs):
        """
        Forward pass for the language model.
        Returns a response object mimicking OpenAI response format.
        """
        # For chat models, use messages; otherwise, use prompt
        if messages:
            prompt = messages[0]["content"] if isinstance(messages, list) and messages else prompt
        response = self._generate(prompt, **kwargs)
        return response

    async def aforward(self, prompt=None, messages=None, **kwargs):
        # Optionally implement async generation if needed
        raise NotImplementedError("Async not implemented for HFModel.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

def _check_api(api_key:str, api_env_name:str):
    if not api_key and not os.environ.get(api_env_name):
        logger.error(f"{api_env_name} must be provided either as an argument or as an environment variable {api_env_name}")
        # raise ValueError(
        #     f"{api_env_name} must be provided either as an argument or as an environment variable {api_env_name}"
        # )
        sys.exit(1)
