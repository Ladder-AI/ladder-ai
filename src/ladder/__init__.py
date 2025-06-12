from ladder.engines import LLMEngine, VerificationEngine
from ladder.utils import setup_default_engines, load_basic_configs
from ladder.data_gen import VLadder, Dataset
from ladder.data_gen.generator import create_dataset_generator
from ladder.finetuning import Ladder, TTRL
from ladder.config import LadderConfig
from dotenv import load_dotenv
from typing import Callable, Optional
from loguru import logger
import dspy 
import os 


load_dotenv()
dspy.disable_logging()


def create_dataset(*,
                    config: LadderConfig,
                    problem_description: str, 
                    dataset_len: int) -> Dataset:
    """ build basic dataset generator and return all required engines / components """
    
    llm_engine, verification_engine, difficulty_engine = setup_default_engines(config)
    dataset_generator = create_dataset_generator(
        llm_engine=llm_engine,
        verification_engine=verification_engine,
        difficulty_engine=difficulty_engine,
    )
    dataset = dataset_generator.generate_dataset(
        problem_description=problem_description,
        initial_problems=[],
        max_dataset_size=dataset_len
    )
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
                   **kwargs
                   ):
    Qtrain, Qtest = vladder_dataset.split(0.8)

    ### Load Engines
    if not verification_engine:
        llm_engine = LLMEngine(lm=config.finetune_llm_runner)
        verification_engine = VerificationEngine(llm_engine=llm_engine)

    ### Ladder
    ladder = Ladder(vladder=Qtrain, config=config, verification_engine=verification_engine, reward_funcs=reward_funcs, **kwargs)
    ladder_tuned_model = ladder.finetune(save_locally=True)     

    return ladder_tuned_model
    
__all__ = ["create_dataset", "load_dataset", "finetune_model", "LadderConfig", "Ladder", "TTRL"]
