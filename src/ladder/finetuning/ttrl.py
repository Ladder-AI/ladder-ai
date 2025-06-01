from typing_extensions import Annotated, Doc, Optional, Any
from ladder.finetuning import FinetuningEngine
from ladder.engines import VerificationEngine
from ladder.data_gen.generator import DatasetGenerator

class TTRL(FinetuningEngine):
    """ Finetuning Engine using TTRL Algorithm
    """

    def __init__(self,
                 *,
                 target_llm: Annotated[Optional[str | Any], Doc(
                     """Target LLM to be finetuned, hf compatible models"""
                 )] = None,
                 verification_engine: Annotated[VerificationEngine, Doc(
                     """Problem Verification Engine"""
                 )],
                 dataset_generator: Annotated[DatasetGenerator, Doc(
                     """Dataset Generator, will be used to generate dataset variants within training loop"""
                 )],
                 tuned_model: Annotated[Any, Doc(
                     """Base Finetuned Model , by ladder for ex"""
                 )] = None
                 ):
        # One of these models should be provided TODO:: verify
        self.target_llm = target_llm
        self.tuned_model = tuned_model

        self.verification_engine = verification_engine
        self.dataset_generator = dataset_generator

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def finetune(self, *args, **kwargs):
        """TODO:: implement TTRL finetuning Algorithm """