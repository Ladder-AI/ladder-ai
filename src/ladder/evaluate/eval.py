from ladder.engines import VerificationEngine , LLMEngine
from ladder.schema import Problem, VLadder
from ladder.llms import BaseLM
from ladder.utils import get_device, generate_hf_model_answer, load_hf_model
from typing import Optional, Union, Literal
from pydantic import BaseModel
from loguru import logger
import torch
import sys
import gc

try:
    from transformers import PreTrainedModel, AutoTokenizer
except ImportError:
    logger.error("transformers not installed. Please install with `pip install transformers`")
    raise

# Result schemas
class EvaluationScore(BaseModel):
    avg_score: float
    all_scores: list[float]
    accuracy: float

class EvaluationResults(BaseModel):
    base_model: Optional[EvaluationScore] = None
    tuned_model: Optional[EvaluationScore] = None


def evaluate_llm(
    q_test: VLadder,
    base_model: BaseLM,
    tuned_model: Optional[Union[PreTrainedModel, str]] = None,
    *args,
    **kwargs
) -> EvaluationResults:
    """ Evaluate LLM Model before and after tuning """
    # Main Evaluation Function
    if not tuned_model:
        logger.error("You must provide a tuned model (as object or path).")
        sys.exit(1)

    results = EvaluationResults()

    verification_engine = VerificationEngine(
        llm_engine=LLMEngine(lm=base_model),
    )

    # Evaluate base model (Before tuning)
    if base_model:
        logger.warning("Evaluating base model...")
        results.base_model = _evaluate_model(base_model, q_test, "base", verification_engine=verification_engine)
        logger.success(f"""Base model \n
                                1. accuracy: {results.base_model.accuracy} \n
                                2. avg_score: {results.base_model.avg_score} \n
                                """)

    # Evaluate tuned model (After tuning)
    logger.warning("Evaluating tuned model...")

    # Load models
    tuned_model = load_hf_model(tuned_model)
    results.tuned_model = _evaluate_model(tuned_model, q_test, "tuned", verification_engine=verification_engine)

    logger.success(f"""Tuned model \n
                            1. accuracy: {results.tuned_model.accuracy} \n
                            2. avg_score: {results.tuned_model.avg_score} \n
                            """)

    return results


# Internal Evaluation Helper
def _evaluate_model(model: Union[PreTrainedModel, BaseLM], 
                    verification_engine: VerificationEngine ,
                    q_test: VLadder,
                    model_type: Literal["base", "tuned"],
                    ) -> EvaluationScore:
    
    q_test.items = q_test.items[0:2]
    logger.debug(f"Starting Evaluation over {len(q_test.items)} problems...")
    
    if model_type == "tuned" and isinstance(model, BaseLM):
        logger.error("You must provide a tuned model (as object or path).")
        sys.exit(1)
    
    if model_type == "base" and isinstance(model, PreTrainedModel):
        logger.error("You must provide a base model (as object or path).")
        sys.exit(1)

    # verifier = VerificationEngine(llm_engine=LLMEngine(lm=model)) # this model shoulddnt be smaller one 
    verifier = verification_engine
    all_scores = []
    correct = 0

    if(not q_test.items):
        logger.error("No items in VLadder")
        return 

    if model_type == "tuned":

        try:
            tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
            device = get_device()
            model.to(device)
            for item in q_test.items:
                answer = generate_hf_model_answer(
                    model=model,
                    question=item.prompt,
                    tokenizer=tokenizer,
                    device=device
                )
                score = verifier.compare_answers(
                    main_question=item.prompt,
                    original_answer=item.completion,
                    generated_answer=answer,
                )
                if score > 0.5:
                    correct += 1
                all_scores.append(score)
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
        finally:
            # Clean up
            model.cpu()  # Move model back to CPU
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    else:
        for item in q_test.items:
            answer = model(item.prompt)
            if len(answer):
                answer = answer[0]
            score = verifier.compare_answers(
                main_question=item.prompt,
                original_answer=item.completion,
                generated_answer=answer,
            )
            if score > 0.5:
                correct += 1
            all_scores.append(score)
       
    accuracy = correct / len(q_test.items)
    avg_score = sum(all_scores) / len(all_scores)
    return EvaluationScore(avg_score=avg_score, all_scores=all_scores, accuracy=accuracy)


