from ladder import setup_default_engines, load_basic_configs, load_dataset, generate_dataset
from ladder.finetuning import Ladder
from huggingface_hub import login
from dotenv import load_dotenv
from loguru import logger
import dspy 
import os 

load_dotenv()
dspy.disable_logging()

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")
# make sure to create .env file and add your openai api key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") 

# login to huggingface hub
login(token=os.environ.get("HF_TOKEN"))


def load_vladder():
    logger.warning("1- Loading dataset...")
    # TODO:: add param to generate_dataset like samlple examples 
    # dataset = generate_dataset(problem_description=problem_description, config=config, dataset_len=10)
    dataset = load_dataset("../data/graph/dataset7.json")
    vladder_dataset = dataset.to_vladder() # or VLadder.from_hf_dataset(dataset)
    vladder_dataset.apply_pattern("Answer: {}")
    logger.success("Dataset loaded successfully")
    return vladder_dataset


# 0- basic config
problem_description = """
                    Title: Balanced Paths in Weighted Directed Graphs 
                    Description: In a directed graph  G=(V,E) with weighted edges, a path is considered balanced if, for every intermediate vertex  v∈V {s,t}, 
                    the sum of the weights of edges entering v along the path equals the sum of the weights of edges leaving  v along the path. The graph may contain arbitrary weights, 
                    including positive, negative, or zero values. The structure and properties of such paths depend on the topology of the graph and the distribution of weights.    
                    """
config = load_basic_configs(push_to_hub=True, hub_model_id="ladder-v1") # LLM > openai/gpt-3.5-turbo, Qwen/Qwen2-0.5B

dataset = generate_dataset(problem_description=problem_description, config=config, dataset_len=3)

# # TODO:: add anthor option for create new dataset from scratch 
# # 1- Load / generate vladder  
# vladder_dataset = load_vladder()
# # split dataset 
# Qtrain, Qtest = vladder_dataset.split(0.8)


# # # 2- (finetune)
# logger.warning("2- Start finetuning")
# _, verification_engine, _ = setup_default_engines(config=config)

# # TODO:: add schema for reward functinos , verification engine override 
# ladder = Ladder(vladder=Qtrain, config=config,verification_engine=verification_engine, reward_funcs=[]) # add custom reward functions as u need 
# finetuned_model = ladder.finetune(save_locally=True)
# logger.success("Model finetuned successfully")

# # 3- export model (make it compatible with HF)
# finetuned_model.push_to_hub()

# Docs 

# git tag -a v0.1.3 -m "Release version 0.1.3"
# git push origin v0.1.3