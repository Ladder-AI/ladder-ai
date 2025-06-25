from ladder.data_gen.schema import Problem,Transformation, SubProblem
from typing_extensions import Annotated, Doc 
from ladder.engines import LLMEngine
from typing import Optional
import dspy 

class _ProblemDifficultyAdapter(dspy.Signature):
    """
    You are a problem difficulty adaptation expert specializing in modifying educational problems for controlled model training.

    Adjust the difficulty of a given problem by applying targeted transformations.
    
    Behavior is determined by:
    - `increase_difficulty`: If True, the problem should be made harder; if False, easier.
    - `model_intelligence_ratio`: Optional float (0.0–1.0) indicating the capability of the model; used to calibrate transformation intensity.

    The given transformations is just example but you can adjust the problem as needed 
    example for math problem , you can increase the equation degrees, complexity or make it simpler,harder and so on for any other domains .. .
    Return the updated problem and the list of transformations used.
    """

    problem: Problem = dspy.InputField(
        prefix="Problem: ",
        format=Problem,
        description="Problem to be modified (harder or easier)"
    )

    increase_difficulty: bool = dspy.InputField(
        prefix="Increase Difficulty: ",
        format=bool,
        description="True to increase difficulty, False to decrease it"
    )

    model_intelligence_ratio: Optional[float] = dspy.InputField(
        prefix="Model Intelligence Ratio: ",
        format=float,
        description="Optional difficulty calibration value (0.0–1.0) representing model capability"
    )

    out_problem: Problem = dspy.OutputField(
        prefix="Modified Problem: ",
        format=Problem,
        description="Resulting problem after difficulty adjustment"
    )

    transformations: list[Transformation] = dspy.OutputField(
        prefix="Applied Transformations: ",
        format=list[Transformation],
        description="List of transformations used to adjust the problem’s difficulty"
    )

class _SubProblemDifficultyAdapter(dspy.Signature):
    """
    You are a problem difficulty adaptation expert specializing in fine-grained control of problem difficulty.

    Modify a problem's difficulty using targeted transformations based on:
    - `increase_difficulty`: Direction of difficulty adjustment.
    - `model_intelligence_ratio`: Optional float (0.0–1.0) to guide the degree of change.

    Return the updated problem (mainly new question adjusted to make it easier or harder) and the list of applied transformations.
    """

    problem: SubProblem = dspy.InputField(
        prefix="problem: ",
        format=SubProblem,
        description="problem to be modified (harder or easier)"
    )

    increase_difficulty: bool = dspy.InputField(
        prefix="Increase Difficulty: ",
        format=bool,
        description="True to increase difficulty, False to decrease it"
    )

    model_intelligence_ratio: Optional[float] = dspy.InputField(
        prefix="Model Intelligence Ratio: ",
        format=float,
        description="Optional calibration value (0.0–1.0) representing model’s ability to solve the problem"
    )

    out_subproblem: SubProblem = dspy.OutputField(
        prefix="Modified problem: ",
        format=SubProblem,
        description="Resulting problem after difficulty adjustment"
    )

    transformations: list[Transformation] = dspy.OutputField(
        prefix="Applied Transformations: ",
        format=list[Transformation],
        description="List of transformations used to modify the problem’s difficulty"
    )


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
        out = self.problem_difficulty_adapter(problem=problem,model_intelligence_ratio=model_intelligence_ratio, increase_difficulty=increase_difficulty)
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
        out = self.subproblem_difficulty_adapter(problem=subproblem,model_intelligence_ratio=model_intelligence_ratio, increase_difficulty=increase_difficulty)
        return out.out_subproblem, out.transformations