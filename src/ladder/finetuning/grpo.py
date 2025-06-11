from ladder.schema import VLadder
from typing import Callable, Optional
from ladder.config import LadderConfig
from loguru import logger
try:
    from trl import GRPOTrainer, GRPOConfig
    from transformers import  PreTrainedModel
except ImportError:
    logger.error("trl is not installed. Please install trl using `pip install trl transformers`")
    raise ImportError("trl is not installed. Please install trl using `pip install trl transformers`")

def _prepare_grpo_config(grpo_config: Optional[GRPOConfig],config: LadderConfig) -> GRPOConfig:
    """Prepare GRPO Configuration for training"""
    return grpo_config or GRPOConfig(
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        push_to_hub=config.push_to_hub
    )

def grpo(vladder:VLadder, 
         model:PreTrainedModel, 
         config:LadderConfig,
         grpo_config:Optional[GRPOConfig]=None, 
         reward_funcs:list[Callable] = []) -> PreTrainedModel:
    
    grpo_config = _prepare_grpo_config(grpo_config=grpo_config,config=config)
    # Convert VLADDER dataset to Hugging Face Dataset format
    vladder = vladder.to_hf_dataset()

    # Initialize the GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=vladder,
        reward_funcs=reward_funcs,
    )
    # Start training
    trainer.train()

    return model