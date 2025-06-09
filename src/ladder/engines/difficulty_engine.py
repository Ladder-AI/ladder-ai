from ladder.data_gen.schema import Problem,Transformation, SubProblem
from typing_extensions import Annotated, Doc 
from ladder.engines import LLMEngine
from typing import Optional
import dspy 

class _ProblemDifficultyAdapter(dspy.Signature):
    """ utils to make the problem harder or easier 
        according to 
        - increase_difficulty (will decide either to increase or decrease difficulty)
        - model_intelligence_ratio if given will decide how much the problem should be changed
    """
    problem: Problem  = dspy.InputField(prefix="problem: ", 
                                       format=Problem , 
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
    
    # TODO:: this should be input ?? 
    transformations: list[Transformation] = dspy.OutputField(prefix="transformations: ", 
                                                  format=list[Transformation], 
                                                  description="List of transformation(s) used to change the problem difficulty")


class _SubProblemDifficultyAdapter(dspy.Signature):
    """ This Engine will be used to change the problem difficulty, estimate the difficulty levels 
    """
    subproblem: SubProblem  = dspy.InputField(prefix="subproblem: ", 
                                       format=SubProblem , 
                                       description="SubProblem to be made harder")
    
    increase_difficulty: bool = dspy.InputField(prefix="increase_difficulty: ", 
                                                format=bool, 
                                                description="Whether to increase difficulty or not")
    
    model_intelligence_ratio: Optional[float] = dspy.InputField(prefix="model_intelligence_ratio: ", 
                                                               format=int, 
                                                               description="decide what difficulty level the model can solve")

    out_subproblem: SubProblem = dspy.OutputField(prefix="harder_subproblem: ", 
                                               format=SubProblem, 
                                               description="Harder SubProblem")
    
    # TODO:: this should be input ?? 
    transformations: list[Transformation] = dspy.OutputField(prefix="transformations: ", 
                                                  format=list[Transformation], 
                                                  description="List of transformation(s) used to change the subproblem difficulty")

class DifficultyEngine(dspy.Module):
    """ This Engine will be used to change the problem difficulty, estimate the difficulty levels 
    """

    problem_difficulty_adapter = dspy.ChainOfThought(_ProblemDifficultyAdapter)
    subproblem_difficulty_adapter = dspy.ChainOfThought(_SubProblemDifficultyAdapter)

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
        
        # TODO:: add anthor version for subproblem too 
    
    def change_subproblem_difficulty(self,
                                    subproblem: SubProblem,
                                    model_intelligence_ratio: Optional[float]=None,
                                   increase_difficulty: bool=True) -> tuple[SubProblem,Transformation]:
        """ Make the subproblem harder or easier
        
        Returns:
            - subproblem: Harder / Easier generated subproblem 
            - transformations: List of transformation(s) used to change the subproblem difficulty            
        """
        out = self.subproblem_difficulty_adapter(subproblem=subproblem,model_intelligence_ratio=model_intelligence_ratio, increase_difficulty=increase_difficulty)
        return out.out_subproblem, out.transformations