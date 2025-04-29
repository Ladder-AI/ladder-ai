from ladder.engines.llm_engine import LLMEngine
from ladder.data_gen.schema import Problem
from typing import Callable
import dspy


class ProblemSolutionVerifier(dspy.Signature):
    problem: str = dspy.InputField(prefix="problem: ", 
                                       format=str, 
                                       desc="Problem to be verified")
    solution: str = dspy.InputField(prefix="solution: ",
                                        format=str,
                                        desc="LLM Solution to the problem")
    
    result: float = dspy.OutputField(prefix="result: ",
                                     format=float,
                                     decs="0 if the solution is incorrect and 1 if it is surely correct")


class VerificationEngine(dspy.Module):
    """Problem Verification Engine

    Verifies whether the LLM-generated solution is correct.
    Used during dataset generation and fine-tuning processes.
    """
    def __init__(self, 
                *, 
                llm_engine:LLMEngine, 
                callbacks: list[Callable]=None):
        super().__init__() 
        self.llm_engine = llm_engine
        self.callbacks = callbacks

        self.problem_solution_verifier = dspy.ChainOfThought(ProblemSolutionVerifier)

    def verify(self, problem: Problem) -> float:
        """Automated verification of LLM Solution

        Should return:
        - 1.0 if the solution is correct
        - 0.0 if it is incorrect

        in this base class we will be using the llm_engine to verify the solution , but u can override this for custom verification
        """
        return self.problem_solution_verifier(problem=problem.question, answer=problem.answer).result
        
        