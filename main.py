from ladder.dataset import DatasetGenerator, VerificationEngine
from ladder.llms import LLMEngine, FinetunedLLMType, LLMType
from ladder.finetuning import Ladder, TTRL
from ladder.utils import load_json
from typing import Any 

# 0- Setup Dependencies
base_inference_llm = LLMType.GPT_4_TURBO # This LLM will be used during dataset generation
base_finetune_llm = FinetunedLLMType.LLAMA_2_13B_CHAT # This LLM will be used during finetuning
problem_description = """ Define the problem description here"""

llm_engine = LLMEngine(base_llm=base_inference_llm)
verification_engine = VerificationEngine(llm_engine=llm_engine) # LLM That will be used in dataset generation could be larger than llm used in finetuning
dataset_generator = DatasetGenerator(
        llm_engine=llm_engine,
        verification_engine=verification_engine
    )

def ladder(dataset_path: str = None):
    """ Ladder Algorithm"""
    
    # 1- Generate Dataset (3 steps, verification engine, difficulty engine)
    if not dataset_path:
        vladder_dataset = dataset_generator.generate_dataset(problem_description=problem_description)
        vladder_dataset.to_json() # export dataset
    else:
        vladder_dataset = load_json(dataset_path)

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
    ladder_finetuned_model = ladder() # pass dataset path if already generated
    ttrl_finetuned_model = ttrl(model=ladder_finetuned_model)

    # TODO:: 
    # 1- Verification and Benchmarking
    # 2- Export Finetuned Models

