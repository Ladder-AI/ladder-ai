from ladder.engines import LM, LLMEngine, VerificationEngine, DifficultyEngine
from ladder.data_gen.generator import DatasetGenerator, Dataset
from ladder.config import LadderConfig
from ladder.engines import LM
from loguru import logger
import os 

# 1- define configs 
def load_basic_configs(hub_model_id="ladder", push_to_hub=True, **kwargs: dict):
    config = LadderConfig(
        finetune_base_llm="Qwen/Qwen2-0.5B",
        inference_base_llm="openai/gpt-3.5-turbo",
        max_steps=3,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        bf16=True,
        output_dir=hub_model_id,
        **kwargs
    )
    return config

def setup_default_engines(config: LadderConfig) -> tuple[LLMEngine, VerificationEngine, DifficultyEngine]:
    """ setup basic required engines for dataset generation process and ladder finetuning"""

    llm_engine = LLMEngine(lm=config.inference_base_llm)

    verification_engine = (
        VerificationEngine(llm_engine=llm_engine) 
    )
    difficulty_engine = (
        DifficultyEngine(llm_engine=llm_engine)
    )
    return llm_engine, verification_engine, difficulty_engine


def generate_dataset(*,
                    config: LadderConfig,
                    problem_description: str, 
                    dataset_len: int) -> Dataset:
    """ build basic dataset generator and return all required engines / components """
    
    llm_engine, verification_engine, difficulty_engine = setup_default_engines(config)
    dataset_generator = DatasetGenerator(
        problem_description=problem_description,
        llm_engine=llm_engine,
        verification_engine=verification_engine,
        difficulty_engine=difficulty_engine,
        max_dataset_to_generate=dataset_len
    )
    dataset = dataset_generator.forward()
    return dataset

def load_dataset(dataset_path:str):
    """ load dataset from json """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError
    return Dataset.from_json(dataset_path)

