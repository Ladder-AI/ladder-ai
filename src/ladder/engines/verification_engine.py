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
    # TODO:: this should be small llm to be finetuned not the large one 
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
        
        
    
def verification_reward_factory(engine: VerificationEngine) -> Callable[[list[str], list[str]], list[float]]:
    """
    Factory that produces a reward function from a VerificationEngine instance.
    
    This function wraps the engine.verify method and makes it compatible
    with the GRPO reward function format.
    """
    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Wrap into a Problem schema (adjust as needed)
            problem = Problem(question=prompt, answer=completion)
            score = engine.verify(problem)
            rewards.append(score)
        return rewards

    return reward_func