
"""
List of Steps to generate Dataset for Ladder Finetuning
"""
from ladder.data_gen.schema import SubProblem, Problem
from typing_extensions import Doc, Annotated
from ladder.engines import  DifficultyEngine
from ladder.engines import LLMEngine
import dspy 


class ProblemVerification(dspy.Signature):
    """PRESTEP1:: Verify if the problem is suitable for Ladder Finetuning

    For the problem to be suitable for Ladder Finetuning, it should meet the following criteria:
        1. Verification process could be automated (dummy as we can pass verification engine as a proof)
        2. Problem Could have multiple difficulty Level
        3. Variants Generation is Possible for each problem (one can generate different variants for the same problem)
    """
    problem_description: str = dspy.InputField(prefix="problem_description: ", 
                                               format=str, 
                                               desc="A string containing the problem description, from which the transformations will be defined")
    
    
    description : str = dspy.OutputField(format=str, 
                                                   desc="A string explaining why this problem is suitable or not for Ladder Finetuning")
    is_ladder_suitable: bool = dspy.OutputField(format=bool, 
                                          desc="Boolean value that indicates if the problem is suitable for Ladder Finetuning")
    

class ProblemGenerator(dspy.Signature):
    """PRESTEP2:: used in dataset initalization process to generate new problem if required """
    problem_description: str = dspy.InputField(prefix="problem_description: ", 
                                               format=str, 
                                               desc="A string containing the problem description, from which we have to define new problem , with its solution , ..")
    
    new_problem: Problem = dspy.OutputField(format=Problem, 
                                            desc="new generated dataset item / problem that will be used later in the finetuning process (only question, answer and difficulty level)")
    

class TransformationLibrary(dspy.Signature):
    """
    STEP1:: You are given a problem description and your job is to generate list of transformations 
    that will be used in the variants generation process. This should be considered as the first step 
    in the dataset generation process.
    """
    problem_description: str = dspy.InputField(prefix="problem_description: ", 
                                               format=str, 
                                               desc="A string containing the problem description, from which the transformations will be defined")
    # example_problem: Optional[Problem] = dspy.InputField(format=Problem, 
    #                                            desc="A string containing an example problem which the model failed to solve")

    model_intelligence_ratio: float = dspy.InputField(prefix="model_intelligence_ratio: ",
                                                       format=float,
                                                       desc="decide the difficulty threshold after which the model cant solve the problem")
    
    make_easier : bool = dspy.InputField(prefix="make_easier: ",
                                         format=bool,
                                         desc="whether to generate transformations that make the problem easier or not")
    transformations: list[str] = dspy.OutputField(format=list[str], 
                                                desc="""List of transformation strings that will be used in the variants generation process. 
                                                Each string should be formatted as: " <transformation_description> || <difficulty_level>"
                                                where <difficulty_level> is a float indicating the transformation's difficulty. If this value is larger than 
                                                the model_intelligence_ratio, it means this transformation makes the problem harder, and vice versa.
                                                For example: "Add time constraints to the problem || 0.8" """)
    
    # TODO:: Transoforamtion now is generated from the problem description (fixed step outside loop) later this should be anthor reasoning task
    # that should generate required description according to specific task or problem (not general description)

class VariantGenerator(dspy.Signature):
    """
    STEP2:: You are given a specific problem / task and your job is to generate list of variants of this problem 
    by randomly applying transformations to the problem description. This should be considered as the second step 
    in the dataset generation process.
    """
    transformations: list[str] = dspy.InputField(prefix="transformations: ", 
                                                 format=list[str], 
                                                 desc="List of transformations that will be used in the variants generation process")
    problem: Problem = dspy.InputField(format=Problem, 
                                       desc="A problem from which we should generate new variants")
    
    variants: list[Problem] = dspy.OutputField(format=list[str], 
                                           desc="List of new variants / problems generated by applying transformations to the problem")


class RecursiveVariantsTree(dspy.Signature):
    """
    STEP3:: You are given a specific problem / task and your job is to generate list of variants of this problem 
    by recursively applying transformations to the problem description to make it more easier up to N difficulty Levels. This should be considered as the third step 
    in the dataset generation process.
    """
    transformations: list[str] = dspy.InputField(prefix="transformations: ", 
                                                 format=list[str], 
                                                 desc="List of transformations that will be used in the variants generation process")
    
    problem: Problem = dspy.InputField(format=Problem,
                                       desc="A problem from which we should generate new subproblems which should be easier for the model to solve")
    
    # TOOD:: maybe we need here too the model intelligence level
    n: int = dspy.InputField(prefix="N: ", 
                             format=int, 
                             desc="Number of difficulty levels to go  down") # TODO:: this should be generated dynamically from the difficulty Engine
    sub_variants: list[SubProblem] = dspy.OutputField(format=list[str], 
                                           desc="List of variants/ subproblems that will be used later as the core part for Ladder finetuning process")

    # during this step we need to identify too if the problem is solvable or not and its difficulty level
    # (this should be difficulty engine task)


class _SubProblemTester:
    """Step3_SUB helper

    Generate List of subproblems according to the model intelligence level (in step3)
    """
    # TODO:: Work here 
    # TODO:: see how to select small llm and what framework to use and how to make it adapter to other frameworks
    def __init__(self, llm_engine: Annotated[LLMEngine, Doc("LLM Engine to be used for dataset generation")], difficulty_engine: DifficultyEngine):
        self.llm_engine = llm_engine
        self.difficulty_engine = difficulty_engine

    def test_llm_on_subproblem(self, subproblem: SubProblem) -> bool:
        """
        Test the LLM on a given subproblem and return if it solved it correctly.
        """
        response = self.llm_engine.lm(subproblem.sub_question)
        return self.verify_llm_solution(response, subproblem)

    def verify_llm_solution(self, response: str, subproblem: SubProblem) -> bool:
        """
        Check if the LLM's response is correct based on the expected solution.
        This verification can be based on similarity or exact matching of the response.
        """
        # Assuming subproblem has a 'solution' field that holds the expected solution
        # TODO:: can we do this verification using verification_engine
        if isinstance(response, list):
            response = response[0]
        return response.strip() == subproblem.sub_answer.strip()

    def adjust_difficulty(self, subproblem: SubProblem, solve_success: bool) -> SubProblem:
        """
        Adjust the difficulty of the subproblem based on whether the LLM could solve it or not.
        If solved, increase difficulty; if not, decrease difficulty.
        """
        if solve_success:
            # Make the subproblem harder
            new_subproblem, _ = self.difficulty_engine.change_subproblem_difficulty(
                subproblem=subproblem, 
                model_intelligence_ratio=0.8,  # Adjust as necessary
                increase_difficulty=True
            )
        else:
            # Make the subproblem easier
            new_subproblem, _ = self.difficulty_engine.change_subproblem_difficulty(
                subproblem=subproblem, 
                model_intelligence_ratio=0.8,  # Adjust as necessary
                increase_difficulty=False
            )
        return new_subproblem

    def generate_subproblems(self, base_problem: Problem, n: int = 3) -> list[SubProblem]:
        """
        Generate a list of subproblems by recursively adjusting difficulty.
        """
        subproblems = []
        for _ in range(n):
            subproblem = SubProblem(
                sub_question=base_problem.question, 
                sub_answer=base_problem.answer,
                difficulty_level=base_problem.difficulty_level
            )
            solved = self.test_llm_on_subproblem(subproblem)
            adjusted_subproblem = self.adjust_difficulty(subproblem, solved)
            subproblems.append(adjusted_subproblem)
        return subproblems