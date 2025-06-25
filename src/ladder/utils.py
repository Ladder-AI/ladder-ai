from typing import Union, Optional
from loguru import logger
import torch
import json 
import sys
import os 

try:
    from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("transformers not installed. Please install with `pip install transformers`")
    raise

from ladder.engines import LLMEngine, VerificationEngine, DifficultyEngine
from ladder.llms import OllamaModel, OpenAIModel
from ladder.config import LadderConfig

# 1- define configs 
def load_basic_configs(hub_model_id="ladder", push_to_hub=False, **kwargs: dict):
    config = LadderConfig(
        finetune_llm_runner=OllamaModel(model="llama3.2:latest"),
        instructor_llm=OpenAIModel(model="openai/gpt-3.5-turbo"),
        target_finetune_llm_id="meta-llama/Llama-3.2-1B",
        max_steps=3,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        bf16=True,
        output_dir=hub_model_id,
        report_to=None,
        **kwargs
    )
    return config

def setup_default_engines(config: LadderConfig) -> tuple[LLMEngine, VerificationEngine, DifficultyEngine]:
    """ setup basic required engines for dataset generation process and ladder finetuning"""

    llm_engine = LLMEngine(lm=config.instructor_llm)
    # ver_llm_engine = LLMEngine(lm=config.finetune_llm_runner)

    verification_engine = (
        VerificationEngine(llm_engine=llm_engine) 
    )
    difficulty_engine = (
        DifficultyEngine(llm_engine=llm_engine)
    )
    return llm_engine, verification_engine, difficulty_engine


def load_json(json_path:str):
    if not os.path.exists(json_path):
        logger.error(f"Json file {json_path} does not exist")
        sys.exit(1)
    with open(json_path, 'r') as f:
        return json.load(f)

def run_local_hf_model():
    pass 

def get_device() -> str:
    if torch.backends.mps.is_available():
        torch.set_default_device("mps")
        return "mps"
    elif torch.cuda.is_available():
        torch.set_default_device("cuda")
        return "cuda"
    else:
        torch.set_default_device("cpu")
        return "cpu"


def generate_hf_model_answer(model: PreTrainedModel, 
                             question: str, 
                             tokenizer: Optional[AutoTokenizer] = None, 
                             device: Optional[str] = None) -> str:
    """ get answer from hf model """

    if not device:
        device = get_device()
    
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    
    model.to(device)

    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text.split("A:")[-1].strip()
    return answer

def load_hf_model(model: Union[str, PreTrainedModel]) -> PreTrainedModel:
    if isinstance(model, str):
        logger.debug(f"Loading model from path: {model}")
        return AutoModelForCausalLM.from_pretrained(model)
    return model