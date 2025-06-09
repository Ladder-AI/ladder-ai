from ladder.engines import VerificationEngine, verification_reward_factory
from ladder.finetuning.grpo import grpo
from typing import Optional, Union, Callable
from typing_extensions import Annotated, Doc
from datasets import Dataset as HFDataset
from ladder.config import LadderConfig
from ladder.data_gen import VLadder
from loguru import logger
import os 

# Check if trl is installed
try:
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
except ImportError:
    logger.error("trl is not installed. Please install trl using `pip install trl transformers`")
    raise ImportError("trl is not installed. Please install trl using `pip install trl transformers`")

# RewardFunc can be a string, model, or callable that accepts two lists and returns a list of floats.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class Ladder:
    """Finetuning Engine using Ladder Algorithm."""

    def __init__(self,
                 vladder: Annotated[VLadder, Doc("Ladder format Dataset to be used for finetuning")], 
                 config: LadderConfig,
                 reward_funcs: list[Callable],
                 verification_engine: Optional[VerificationEngine] = None,
                 grpo_config: Optional[GRPOConfig] = None,
                 out_model_path: Optional[str] = None,
                 completion_pattern: Annotated[str, Doc("""Completion pattern to be applied on the completions""")] = None,
                 *args, **kwargs):
        """
        Initializes the Ladder finetuning engine using the Ladder Algorithm.

        Args:
            vladder (VLadder): Dataset in Ladder format.
            config (LadderConfig): Configuration settings for the finetuning process.
            base_llm (str): Name or path to the base LLM.
            reward_funcs (list[Callable]): List of reward functions for GRPO trianing process
            grpo_config (Optional[GRPOConfig]): GRPO-specific configuration.
            out_model_path (Optional[str]): Path to save the finetuned model.
        """
        self.base_llm = config.target_finetune_llm_id
        self._config = config
        
        self.completion_pattern = completion_pattern
        self.vladder = vladder if not completion_pattern else vladder.apply_pattern(completion_pattern)
        self.verification_engine = verification_engine
        self.reward_funcs = self._prepare_reward_funcs(reward_funcs)
        self.out_model_path = out_model_path

        # Load tokenizer, configs and model
        self.grpo_config = grpo_config

        logger.warning(f"HF_TOKEN: {os.environ.get('HF_TOKEN')}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_llm)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(self.base_llm)
    


    def _to_hf_dataset(self) -> HFDataset:
        """ Convert VLADDER dataset to Hugging Face Dataset format """
        # Assumes VLadder class has a method to convert to HF dataset
        return self.vladder.to_hf_dataset()

    
    def _prepare_reward_funcs(self,reward_funcs: list[Callable]) -> list[Callable]:
        """Prepare reward functions for GRPO training.

        This method appends extra reward functions based on configuration,
        including a completion pattern check and LLM-based verification.

        Returns:
            list[Callable]: Final list of reward functions.
        """
        final_rewards = list(reward_funcs or [])

        # Pattern-based reward
        if self._config.apply_reward_completion_pattern:
            def pattern_check_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
                pattern = getattr(self._config, "reward_completion_pattern", "Answer:")
                return [1.0 if completion.strip().startswith(pattern) else 0.0 for completion in completions]

            final_rewards.append(pattern_check_reward)

        # Verification-based reward
        if self._config.apply_verification_reward:
            if not self.verification_engine or not isinstance(self.verification_engine, VerificationEngine):
                raise ValueError("VerificationEngine must be attached as `self.verification_engine` to use verification rewards.")
            final_rewards.append(verification_reward_factory(self.verification_engine))

        return final_rewards

        
    def finetune(self,save_locally: bool = True, *args, **kwargs) -> PreTrainedModel:
        """Finetune the model using the Ladder Algorithm and GRPO"""

        # # Convert VLADDER dataset to Hugging Face Dataset format
        # vladder = self._to_hf_dataset()

        # # Initialize the GRPOTrainer
        # trainer = GRPOTrainer(
        #     model=self.model,
        #     args=self.grpo_config,
        #     train_dataset=vladder,
        #     reward_funcs=self.reward_funcs,
        # )

        # # Start training
        # trainer.train()

        self.model = grpo(self.vladder, 
                          self.model,
                          self._config, 
                          self.grpo_config, 
                          self.reward_funcs)

        # Save the finetuned model after training
        if save_locally:
            self.model.save_pretrained(self.out_model_path or "ladder_finetuned_model")
        return self.model
