from ladder.use_cases.graph import generate_or_load_dataset, ladder, ttrl
import dspy 

dspy.disable_logging()

# TODO:: add configs here like llm and other hyper paramaters
dataset = generate_or_load_dataset(dataset_path="dataset.json", force_regenerate=False)