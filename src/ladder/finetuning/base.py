from abc import ABC, abstractmethod

class FinetuningEngine(ABC):
    """
    Base class for finetuning engines. 

    will be used mainly for ladder , TTRL Algorithms , Later could be extended for other finetuning algorithms
    """
    # TODO:: how to make it compatible with HF TRL trainer

    def __init__(self, 
                 base_llm):
        self.base_llm = base_llm

    # @abstractmethod
    # def __call__(self, *args, **kwargs):
    #     raise NotImplementedError
    
    @abstractmethod
    def finetune(self, *args, **kwargs):
        raise NotImplementedError
    
