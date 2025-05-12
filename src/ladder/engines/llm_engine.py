from typing_extensions import Doc, Annotated
import dspy 


class LM(dspy.LM):
    """
    A Language Model that will be used for inference
    """

class LLMEngine:
    """ LLM Service
    
    will be used during different processes , from dataset generation , and some other automated action during training, TTFT

    - LLM inference 
    - temp cycling  
    - persona based prompting
    """

    def __init__(self, 
                 *,
                 lm: Annotated[ str, Doc("""Language Model to be used for inference""")]) -> None:
        self.lm = dspy.LM(lm)
        dspy.configure(lm=self.lm)
    

    # TODO:: complete these methods 
    def temperature_cycling(self):
        ...
    
    def persona_based_prompting(self):
        ...