from ladder.engines import  FinetunedLLMType
from ladder.finetuning import FinetuningEngine

class Ladder(FinetuningEngine):
    """ Finetuning Engine using Ladder Algorithm """

    def __init__(self,vladder: dict, base_llm: FinetunedLLMType, *args, **kwargs):
        self.vladder = vladder
        self.base_llm = base_llm

    def __call__(self, *args, **kwargs):
        # TODO:: recheck this design if we gonnna need to call this class 
        return super().__call__(*args, **kwargs)

    def finetune(self, *args, **kwargs):
        """implement Ladder finetuning process here """