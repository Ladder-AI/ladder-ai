from typing_extensions import Doc, Annotated
from enum import Enum
import dspy 


class FinetunedLLMType(str, Enum):
    """ List of LLM Types that will be used during the finetuning process """
    LLAMA_2_7B_CHAT = "llama-2-7b-chat-hf"
    LLAMA_2_13B_CHAT = "llama-2-13b-chat-hf"
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct"
    MIXTRAL_8X7B = "mixtral-8x7b"
    GPT_4_TURBO = "gpt-4-turbo"


class LM(dspy.LM):
    """
    A Language Model that will be used for inference
    """


class LLMEngine:
    """ LLM Service
    
    will be used during different processes , from dataset generation , and some other automated action during training, TTFT

    - LLM inference 
    - temp cycling  
    - persona based prompting
    """

    def __init__(self, 
                 *,
                 lm: Annotated[LM, Doc("""Language Model to be used for inference""")]) -> None:
        self.lm = lm
    

    def temperature_cycling(self):
        ...
    
    def persona_based_prompting(self):
        ...