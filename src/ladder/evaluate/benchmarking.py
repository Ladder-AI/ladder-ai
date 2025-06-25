
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from ladder import VLadder, VerificationEngine, LLMEngine
from ladder.utils import generate_hf_model_answer
from typing import Union, List, Optional
from ladder.utils import get_device
from loguru import logger
import pandas as pd
import time


def benchmark_llms(
    q_test: VLadder,
    models: List[Union[str, PreTrainedModel]],
    model_names: Optional[List[str]] = None,
    save_csv_path: Optional[str] = None
) -> pd.DataFrame:
    # TODO:: custom benchmarking schema 
    """
    Evaluate and benchmark multiple LLMs on VLadder dataset.

    Args:
        q_test: VLadder test dataset.
        models: List of models (either paths or PreTrainedModel objects).
        model_names: Optional names corresponding to models.
        save_csv_path: Optional path to save CSV report.

    Returns:
        DataFrame of benchmarking results.
    """
    results = []
    device = get_device()

    for i, model_ref in enumerate(models):
        model_name = model_names[i] if model_names and i < len(model_names) else str(model_ref)

        # Load model if string path
        if isinstance(model_ref, str):
            logger.debug(f"Loading model from path: {model_ref}")
            model = AutoModelForCausalLM.from_pretrained(model_ref).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_ref)
        else:
            model = model_ref.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_ref.name_or_path)

        logger.debug(f"Evaluating model: {model_name}")

        verifier = VerificationEngine(llm_engine=LLMEngine(lm=model))
        scores = []
        correct = 0
        total = len(q_test.items)
        start_time = time.time()
        completions_log = []

        for item in q_test.items:
            # Score via verification engine
            score = verifier.verify(item.prompt)
            scores.append(score)

            # Generate answer
            generated = generate_hf_model_answer(model, tokenizer, item.prompt, device)
            expected = (item.completion or "").strip().lower()

            match = generated.strip().lower() == expected
            if match:
                correct += 1

            completions_log.append({
                "model": model_name,
                "prompt": item.prompt,
                "expected": expected,
                "generated": generated,
                "match": match,
                "score": score
            })

        end_time = time.time()
        avg_score = sum(scores) / total
        accuracy = correct / total
        latency = (end_time - start_time) / total

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "avg_score": avg_score,
            "latency_sec": latency
        })

        logger.success(f"{model_name}: Accuracy={accuracy:.3f}, Score={avg_score:.3f}, Latency={latency:.2f}s")

    df = pd.DataFrame(results)

    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
        logger.debug(f"Saved benchmark results to {save_csv_path}")

    return df


