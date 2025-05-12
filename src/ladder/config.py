from pydantic import BaseModel, Field
from ladder.engines import LM
from typing_extensions import Annotated, Doc 
from transformers import TrainingArguments
from typing import Optional


class LadderConfig(BaseModel,TrainingArguments):
    inference_base_llm: Annotated[
        Optional[ str],
        Doc(
            """Base LLM to be used for general inference like dataset generation, default is openai/gpt-3.5-turbo"""
        ),
    ] = "openai/gpt-3.5-turbo"

    finetune_base_llm : Annotated[
        str,
        Doc(
            """Base LLM to be used for finetuning, hf compatible models"""
        ),
    ] 
    force_regenerate: Optional[bool] = Field(
        default=False,
        description="If True, regenerate dataset even if it exists."
    )

    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to save the dataset (if exist and force_regenerate=False, will skip dataset generation)"
    )

    apply_reward_completion_pattern: Optional[bool] = Field(
        default=True,
        description="If True, will add a reward func based on completion pattern"
    )
    apply_verification_reward: Optional[bool] = Field(
        default=True,
        description="If True, will add a reward func based on verification answer"
    )
    include_for_metrics: Optional[list[str]] = Field(
        default=None,
        description="List of fields to include in metrics"
    )

    lr_scheduler_kwargs: Optional[dict] = Field(
        default=None,
        description="Hyperparameters for learning rate scheduler"
    )
    # TODO:: add more required hyper parameters