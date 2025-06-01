from ladder.llms import BaseLM, OpenAIModel
from typing_extensions import Annotated, Doc 
from transformers import TrainingArguments
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class LadderConfig(BaseModel,TrainingArguments):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructor_llm: Annotated[
        Optional[ BaseLM],
        Doc(
            """Base LLM to be used for general inference like dataset generation, default is openai/gpt-3.5-turbo"""
        ),
    ] = OpenAIModel("gpt-3.5-turbo")

    target_finetune_llm : Annotated[
        Optional[str | BaseLM],
        Doc(
            """Base LLM to be used for finetuning, hf compatible models"""
        ),
    ] = None # TODO:: this shouldnt be optional 
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