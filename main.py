from ladder.dataset import DatasetGenerator, VerificationEngine
from ladder.llms import LLMEngine, FinetunedLLMType, LLMType
from ladder.finetuning import Ladder, TTRL
from ladder.utils import load_json
from loguru import logger
from typing import Any 
import os 

# 0- Setup Dependencies
base_inference_llm = LLMType.GPT_4_TURBO # This LLM will be used during dataset generation
base_finetune_llm = FinetunedLLMType.LLAMA_2_13B_CHAT # This LLM will be used during finetuning
problem_description = """ Define the problem description here"""

llm_engine = LLMEngine(base_llm=base_inference_llm)
verification_engine = VerificationEngine(llm_engine=llm_engine) # LLM That will be used in dataset generation could be larger than llm used in finetuning
dataset_generator = DatasetGenerator(
        problem_description=problem_description,
        llm_engine=llm_engine,
        verification_engine=verification_engine
    )

def generate_dataset(dataset_path:str, force_regenerate: bool = False):

    if not dataset_path:
        dataset_path = "dataset.json"

    if not force_regenerate and os.path.exists(dataset_path):
        logger.warning(f"Dataset already exists at {dataset_path}. Skipping dataset generation")
        logger.info("Use force_regenerate=True if you want to regenerate the dataset")
        return load_json(dataset_path)

    dataset = dataset_generator.generate_dataset()
    dataset.to_json("dataset.json")
    logger.success(f"Dataset generated at {dataset_path}")
    return dataset

def ladder(dataset_path: str = None):
    """ Ladder Algorithm"""
    
    # 1- Generate Dataset (3 steps, verification engine, difficulty engine)
    vladder_dataset = generate_dataset(dataset_path=dataset_path)

    ## 2- Finetuning - Ladder
    ladder = Ladder(vladder=vladder_dataset, base_llm=base_finetune_llm)
    finetuned_model = ladder.finetune()

    return finetuned_model


def ttrl(model: Any = None, base_llm: FinetunedLLMType = None):
    """TTRL Algorithm
    
    Args:
    model: if provided it will be used as base model for finetuning 
    base_llm: if model is None , it will be used as base model for finetuning
    """
    ## 3- TTRL 
    ttrl_enigne = TTRL(
        tuned_model=model,
        base_llm=base_llm,
        verification_engine=verification_engine,
        dataset_generator=dataset_generator
    )
    ladder_ttrl_model = ttrl_enigne.finetune()
    return ladder_ttrl_model


if __name__ == "__main__":
    # 1- generate dataset 
    ## TODO:: 
    ## 1- How to design dataset schema from description 
    ## 2- LLM Engine using ollama/ gemni/ openai (langchain)
    ## 3- Verification Engine (how to verify LLM Solution)
    ## 4- How to generate a list of problems which small LLMS cant solve but large LLMS can (use the verification engine)
    ## 5- Implement the Difficulty Engine
    ## 6- implement dataset generation (3 steps, verification engine, difficulty engine)
    ## 7- check Dspy, Langraph, Langchain , smolagents, crewAI, Autogen
    ## 8- check Distilabel

    dataset = generate_dataset(dataset_path="dataset.json", force_regenerate=False)

    # 2- Ladder 
    # ladder_finetuned_model = ladder(dataset_path="dataset.json")

    # 3- TTRL 
    # ttrl_finetuned_model = ttrl(model=ladder_finetuned_model)

    # TODO:: 
    # 1- Verification and Benchmarking
    # 2- Export Finetuned Models

