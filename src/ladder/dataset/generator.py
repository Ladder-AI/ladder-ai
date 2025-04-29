from .verification import VerificationEngine
from typing_extensions import Annotated, Doc
from pydantic import BaseModel
from ladder.llms import LLMEngine


class DatasetGenerator:
    """ Generate required dataset for specific problem

    will implement these 3 steps (chains) to generate dataset
    1. Transformations
    2. Variants Generation
    3. Recursive Variants Tree 
    """

    def __init__(self,*,
                    problem_description: Annotated[str, Doc(
                    """A string containing the problem description, from which the transformations will be defined"""
                    )],
                    llm_engine: Annotated[LLMEngine, Doc(
                    """LLM Engine to be used for dataset generation"""
                    )],
                    verification_engine: Annotated[VerificationEngine, Doc(
                    """Problem Verification Engine"""
                    )]
                 ):
        self.problem_description = problem_description
        self.llm_engine = llm_engine
        self.verification_engine = verification_engine

        # TODO:: define hyper parameters
        self._transformations = []


    @property
    def transformations(self):
        return self._transformations

    def generate_dataset(self) -> "Dataset":
        # TODO::
        """
        1. Define dataset schema from problem description 
        2. implement main 3 steps (chains)
        """

    def generate_transformations(self) -> list[str]:
        """STEP1:: generate List of transformations that will be used in the variants generation process"""

    def generate_variants(self):
        """STEP2:: generate List of variants that will be used in the variants generation process"""
    
    def generate_recursive_variants_tree(self):
        """STEP3:: generate List of variants that will be used in the variants generation process"""

    def difficulty_engine(self):
        """ utils to estimate the required difficulty level required during variant generation"""

    

class Dataset(BaseModel):
    """Dataset Schema"""
   
    def to_json(self) -> str:
            """ export dataset to json """
            return ""