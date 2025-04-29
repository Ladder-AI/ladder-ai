from enum import Enum


class FinetunedLLMType(str, Enum):
    """ List of LLM Types that will be used during the finetuning process """
    LLAMA_2_7B_CHAT = "llama-2-7b-chat-hf"
    LLAMA_2_13B_CHAT = "llama-2-13b-chat-hf"
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct"
    MIXTRAL_8X7B = "mixtral-8x7b"
    GPT_4_TURBO = "gpt-4-turbo"

class LLMType(FinetunedLLMType):
    """ any LLM Type , for any general inference process """
    MIXTRAL_8X7B = "mixtral-8x7b"
    GPT_4_TURBO = "gpt-4-turbo"

