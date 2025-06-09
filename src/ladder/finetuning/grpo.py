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

def _prepare_grpo_config(self,grpo_config: Optional[GRPOConfig],_config: LadderConfig) -> GRPOConfig:
    """Prepare GRPO Configuration for training"""
    return grpo_config or GRPOConfig(
        learning_rate=_config.learning_rate,
        num_train_epochs=_config.num_train_epochs,
        per_device_train_batch_size=_config.train_batch_size,
        max_steps=_config.max_steps,
        gradient_accumulation_steps=_config.gradient_accumulation_steps,
        push_to_hub=_config.push_to_hub
    )

def grpo(vladder:VLadder, 
         model:PreTrainedModel, 
         config:LadderConfig,
         grpo_config:Optional[GRPOConfig]=None, 
         reward_funcs:list[Callable] = []) -> PreTrainedModel:
    
    grpo_config = _prepare_grpo_config(config,grpo_config)
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