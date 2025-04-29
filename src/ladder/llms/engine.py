from ladder.llms.schema import LLMType

class LLMEngine:
    """ LLM Service
    
    will be used during different processes , from dataset generation , and some other automated action during training, TTFT

    - LLM inference 
    - temp cycling  
    - persona based prompting
    """

    def __init__(self, base_llm:LLMType) -> None:
        self.base_llm = base_llm