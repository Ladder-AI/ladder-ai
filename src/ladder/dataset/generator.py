from ladder.dataset.schema import Dataset
from .verification import VerificationEngine
from ladder.llms import LLMEngine
from typing_extensions import Annotated, Doc





class DatasetGenerator:
    """ Generate required dataset for specific problem

    will implement these 3 steps (chains) to generate dataset
    1. Transformations
    2. Variants Generation
    3. Recursive Variants Tree 
    """

    def __init__(self,*,
                    llm_engine: Annotated[LLMEngine, Doc(
                    """LLM Engine to be used for dataset generation"""
                    )],
                    verification_engine: Annotated[VerificationEngine, Doc(
                    """Problem Verification Engine"""
                    )]
                 ):
        self.llm_engine = llm_engine
        self.verification_engine = verification_engine

    @staticmethod
    def generate_dataset(*,problem_description: str) -> Dataset:
        return DatasetGenerator()

    