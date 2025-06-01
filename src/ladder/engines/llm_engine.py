from ladder.llms import BaseLM
from typing_extensions import Doc, Annotated
import dspy 


class LLMEngine:
    """ LLM Service
    
    will be used during different processes , from dataset generation , and some other automated action during training, TTFT

    - LLM inference 
    - temp cycling  
    - persona based prompting
    """

    def __init__(self, 
                 *,
                 lm: Annotated[ BaseLM, Doc("""Language Model to be used for inference""")]) -> None:
        self.lm = dspy.LM(lm) if isinstance(lm, str) else lm
        dspy.configure(lm=self.lm)
    

    # TODO:: complete these methods 
    def temperature_cycling(self):
        ...
    
    def persona_based_prompting(self):
        ...