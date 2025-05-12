from ladder.data_gen.schema import Problem,Transformation
from typing_extensions import Annotated, Doc 
from ladder.engines import LLMEngine
from typing import Optional
import dspy 

class DifficultyAdapter(dspy.Signature):
    """ utils to make the problem harder or easier 
        according to 
        - increase_difficulty (will decide either to increase or decrease difficulty)
        - model_intelligence_ratio if given will decide how much the problem should be changed
    """
    problem: Problem = dspy.InputField(prefix="problem: ", 
                                       format=Problem, 
                                       description="Problem to be made harder")
    
    increase_difficulty: bool = dspy.InputField(prefix="increase_difficulty: ", 
                                                format=bool, 
                                                description="Whether to increase difficulty or not")
    
    model_intelligence_ratio: Optional[float] = dspy.InputField(prefix="model_intelligence_ratio: ", 
                                                               format=int, 
                                                               description="decide what difficulty level the model can solve")

    out_problem: Problem = dspy.OutputField(prefix="harder_problem: ", 
                                               format=Problem, 
                                               description="Harder Problem")
    
    transformations: list[Transformation] = dspy.OutputField(prefix="transformations: ", 
                                                  format=list[Transformation], 
                                                  description="List of transformation(s) used to change the problem difficulty")


class DifficultyEngine(dspy.Module):
    """ This Engine will be used to change the problem difficulty, estimate the difficulty levels 
    """

    difficulty_adapter = dspy.ChainOfThought(DifficultyAdapter)

    def __init__(self, 
                 *,
                 llm_engine: Annotated[LLMEngine, Doc(
                     """LLM Engine to be used for dataset generation"""
                 )]):
        self.llm_engine = llm_engine
    

    def change_problem_difficulty(self,
                                    problem: Problem,
                                    model_intelligence_ratio: Optional[float]=None,
                                   increase_difficulty: bool=True) -> tuple[Problem,Transformation]:
        """ Make the problem harder or easier
        
        Returns:
            - problem: Harder / Easier generated problem 
            - transformations: List of transformation(s) used to change the problem difficulty            
        """
        out = self.difficulty_adapter(problem=problem,model_intelligence_ratio=model_intelligence_ratio, increase_difficulty=increase_difficulty)
        return out.out_problem, out.transformations # TODO:: check schema 
