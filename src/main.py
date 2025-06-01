from ladder import  LadderConfig, create_dataset, finetune_model, load_dataset
from ladder.llms import OpenAIModel # , HFModel
from huggingface_hub import login
import os 


login(token=os.environ.get("HF_TOKEN"))

# Step 1 - Create Dataset
problem_description = """
                    Title: Balanced Paths in Weighted Directed Graphs 
                    Description: In a directed graph  G=(V,E) with weighted edges, a path is considered balanced if, for every intermediate vertex  v∈V {s,t}, 
                    the sum of the weights of edges entering v along the path equals the sum of the weights of edges leaving  v along the path. The graph may contain arbitrary weights, 
                    including positive, negative, or zero values. The structure and properties of such paths depend on the topology of the graph and the distribution of weights.    
                    """


config = LadderConfig(
    instructor_llm = OpenAIModel(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY") ),
    target_finetune_llm ="Qwen/Qwen2-7B", 
)

# dataset = create_dataset(problem_description=problem_description, config=config, dataset_len=1)
# vladder_dataset = dataset.to_vladder()

dataset = load_dataset("../data/graph/dataset7.json")
vladder_dataset = dataset.to_vladder()

# Step 2 - Finetune
final_model = finetune_model(
    vladder_dataset=vladder_dataset,
    config=config,
    reward_funcs=[],
    use_ttrl=False
)


# git tag -a v0.1.4 -m "Release version 0.1.4"
# git push origin v0.1.4



#  HFModel(model="Qwen/Qwen2-7B", api_key=os.environ.get("HF_TOKEN")),

# Optional: override default models
# config.set_verification_model(ClaudeModel(model="claude-2"))
# config.set_difficulty_model(ClaudeModel(model="claude-2"))