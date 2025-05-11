"""
    This is our Main Usecase (Graph theory Problems)
"""

from ladder.engines import LLMEngine, VerificationEngine,DifficultyEngine, FinetunedLLMType, LM
from ladder.data_gen.generator import DatasetGenerator
from ladder.finetuning import Ladder, TTRL
from ladder.utils import load_json
from typing import Any , Optional
from dotenv import load_dotenv
from loguru import logger
import os 

load_dotenv()

# 0- Setup Dependencies 

## LLM
# if u wanna use differet model  check here 
## https://dspy.ai/learn/programming/language_models/
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") # put here your openai api key 
base_inference_llm = LM("openai/gpt-3.5-turbo")

## LLM To Tune 
base_finetune_llm = FinetunedLLMType.LLAMA_2_13B_CHAT # This LLM will be used during finetuning

## Problem description 
problem_description = """
Title: Balanced Paths in Weighted Directed Graphs 
Description: In a directed graph  G=(V,E) with weighted edges, a path is considered balanced if, for every intermediate vertex  v∈V {s,t}, 
the sum of the weights of edges entering v along the path equals the sum of the weights of edges leaving  v along the path. The graph may contain arbitrary weights, 
including positive, negative, or zero values. The structure and properties of such paths depend on the topology of the graph and the distribution of weights.    
"""
## Inference LLM that will be used for different processes
llm_engine = LLMEngine(lm=base_inference_llm)

## Verification 
class GraphVerificationEngine(VerificationEngine):
    
    def verify(self, problem):
        # TODO:: override verify method to implement out custom verification process
        return super().verify(problem)
    
    
verification_engine = GraphVerificationEngine(llm_engine=llm_engine) # LLM That will be used in dataset generation could be larger than llm used in finetuning

## Difficulty Engine (optional)
difficulty_engine = DifficultyEngine(llm_engine=llm_engine)

## inital dataset exampe (optional)
inital_problems = []

## Dataset Generator
dataset_generator = DatasetGenerator(
        problem_description=problem_description,
        llm_engine=llm_engine,
        verification_engine=verification_engine,
        difficulty_engine=difficulty_engine,
        inital_problems=inital_problems
    )

def generate_or_load_dataset(dataset_path:str, 
                     force_regenerate: bool = False):
    """
        generate required dataset for ladder finetuning process 

        Args:
            dataset_path (str): Path to save the dataset (if exist and force_regenerate=False, will skip dataset generation)
            num_of_datasets (int, optional): Number of datasets to generate. Defaults to 3.
            force_regenerate (bool, optional): If True, will regenerate the dataset even if it already exists. Defaults to False.
    """

    if not dataset_path:
        dataset_path = "dataset.json"

    if not force_regenerate and os.path.exists(dataset_path):
        logger.warning(f"Dataset already exists at {dataset_path}. Skipping dataset generation")
        logger.info("Use force_regenerate=True if you want to regenerate the dataset")
        return load_json(dataset_path)

    dataset = dataset_generator.forward()
    dataset.to_json(dataset_path)
   
    return dataset 

def ladder(dataset_path: str = None):
    """ Ladder Algorithm"""
    
    # 1- Generate Dataset (3 steps, verification engine, difficulty engine)
    vladder_dataset = generate_or_load_dataset(dataset_path=dataset_path)

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
    
    ## TODO:: 
    ## 3- Verification Engine (how to verify LLM Solution)
    ## 4- How to generate a list of problems which small LLMS cant solve but large LLMS can (use the verification engine)
    ## 5- Implement the Difficulty Engine
    ## 6- implement dataset generation (3 steps, verification engine, difficulty engine)
    ## 7- check Dspy, Langraph, Langchain , smolagents, crewAI, Autogen
    ## 8- check Distilabel
    ## 9- check dspy optimizer, configs

    # 1- generate dataset 
    dataset = generate_or_load_dataset(dataset_path="dataset.json", force_regenerate=False)

    # 2- Ladder 
    # ladder_finetuned_model = ladder(dataset_path="dataset.json")

    # 3- TTRL 
    # ttrl_finetuned_model = ttrl(model=ladder_finetuned_model)

    # 4- Verification & Benchmarking
    # 5- Export Finetuned Models
