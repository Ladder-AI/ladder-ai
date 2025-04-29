from pydantic import BaseModel
from ladder.llms import LLMType, FinetunedLLMType
from typing_extensions import Annotated, Doc 


class Config(BaseModel):
    inference_base_llm: Annotated[
        LLMType,
        Doc(
            """Base LLM to be used for general inference like dataset generation"""
        ),
    ] = LLMType.GPT_4_TURBO

    finetune_base_llm : Annotated[
        FinetunedLLMType,
        Doc(
            """Base LLM to be used for finetuning"""
        ),
    ] = FinetunedLLMType.LLAMA_2_13B_CHAT