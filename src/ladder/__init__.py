from ladder.engines import LLMEngine, VerificationEngine, DifficultyEngine
from ladder.data_gen import VLadder, Dataset
from ladder.data_gen.generator import DatasetGenerator
from ladder.finetuning import Ladder, TTRL
from ladder.config import LadderConfig
from dotenv import load_dotenv
from typing import Callable, Optional
from loguru import logger
import dspy 
import os 

load_dotenv()
dspy.disable_logging()



# 1- define configs 
def load_basic_configs(hub_model_id="ladder", push_to_hub=True, **kwargs: dict):
    config = LadderConfig(
        target_finetune_llm="Qwen/Qwen2-0.5B",
        instructor_llm="openai/gpt-3.5-turbo",
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

    verification_engine = (
        VerificationEngine(llm_engine=llm_engine) 
    )
    difficulty_engine = (
        DifficultyEngine(llm_engine=llm_engine)
    )
    return llm_engine, verification_engine, difficulty_engine


def create_dataset(*,
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


def finetune_model(*,
                   vladder_dataset: VLadder,
                   config: LadderConfig,
                   reward_funcs: list[Callable] = [],
                   verification_engine: Optional[VerificationEngine] = None,
                   use_ttrl: bool = False,
                   **kwargs
                   ):
    Qtrain, _ = vladder_dataset.split(0.8)

    ### Load Engines
    if not verification_engine:
        llm_engine = LLMEngine(lm=config.instructor_llm)
        verification_engine = VerificationEngine(llm_engine=llm_engine)

    ### Ladder
    ladder = Ladder(vladder=Qtrain, config=config, verification_engine=verification_engine, reward_funcs=reward_funcs, **kwargs)
    ladder_tuned_model = ladder.finetune(save_locally=True) 

    if use_ttrl:
        # TODO:: complete 
        ttrl = TTRL(target_llm=ladder_tuned_model) # should that be path to the model 
        final_model = ttrl.finetune(save_locally=True)
        return final_model

    return ladder_tuned_model
    
