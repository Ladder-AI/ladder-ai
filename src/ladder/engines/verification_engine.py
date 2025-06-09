from ladder.engines.llm_engine import LLMEngine
from ladder.data_gen.schema import Problem
from typing import Callable
from loguru import logger
import dspy


class AnswerComparator(dspy.Signature):
    main_question: str = dspy.InputField(prefix="main_question: ", 
                                         format=str, 
                                         desc="Main question of the problem")
    
    original_answer: str = dspy.InputField(prefix="original_answer: ", 
                                           format=str, 
                                           desc="Original answer of the problem")
    
    generated_answer: str = dspy.InputField(prefix="generated_answer: ", 
                                            format=str, 
                                            desc="Generated answer of the problem")
    
    result: float = dspy.OutputField(prefix="result: ",
                                     format=float,
                                     decs="0 if the solution is incorrect and 1 if it is surely correct")


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
        self.answer_comparator = dspy.ChainOfThought(AnswerComparator)

    def verify(self, problem: Problem) -> float:
        """Automated verification of LLM Solution

        Should return:
        - 1.0 if the solution is correct
        - 0.0 if it is incorrect

        in this base class we will be using the llm_engine to verify the solution , but u can override this for custom verification
        """
        with dspy.context(lm=self.llm_engine.lm):
            res =  self.problem_solution_verifier(problem=problem.question, answer=problem.answer)
            return res.result
    
    def compare_answers(self, main_question: str, original_answer: str, generated_answer: str) -> float:
        """ Compare two answers and return a reward value based on the similarity or accuracy of the answers. 
            0: wrong answer
            1: correct answer
        """
        with dspy.context(lm=self.llm_engine.lm):
            res = self.answer_comparator(main_question=main_question, 
                                         original_answer=original_answer, 
                                         generated_answer=generated_answer)
            return res.result
        
    
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