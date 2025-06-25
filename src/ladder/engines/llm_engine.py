from ladder.llms import BaseLM
from typing_extensions import Doc, Annotated, Literal
from abc import ABC, abstractmethod
from loguru import logger
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
                 lm: Annotated[ BaseLM | str, Doc("""Language Model to be used for inference""")]) -> None:
        self.lm = dspy.LM(lm) if isinstance(lm, str) else lm
        dspy.configure(lm=self.lm)
    
    @abstractmethod
    def generate(self, prompt: str, exported_format: Literal["str", "dict", "bool", "tuple", "list", "set"] = "str", *args, **kwargs) -> str:
        prompt = f"""
        You are Helpful assistant. Keep your answer short and to the point. dont show steps unless asked.
        <USER INPUT>
        {prompt}
        
        <YOUR RESPONSE>
        """
        response =  self.lm(prompt, *args, **kwargs)

        if isinstance(response, list):
                response: str = response[0]

        print(f"response: {response}, export format : {exported_format}")

        if exported_format == "str":
            return response.strip()

        # lets implment only tuple for now 
        elif exported_format == "tuple":
            return self._ensure_tuple(response)
        # TODO:: implement other formats
        else:
            raise ValueError(f"Unsupported export format: {exported_format}")

    def _ensure_tuple(self, answer: str | list| tuple) -> tuple:
        """
        Ensure the answer is a tuple format.
        Handle various input formats: tuple, list, string representation, etc.
        """
        try:
            if isinstance(answer, tuple):
                return answer
            elif isinstance(answer, list) and len(answer) == 2:
                return tuple(answer)
            elif isinstance(answer, str):
                # Try to parse string representation of tuple/coordinates
                # Handle formats like "(1.5, 2.3)", "1.5, 2.3", "[1.5, 2.3]"
                clean_str = answer.strip().strip('()[]')
                coords = [float(x.strip()) for x in clean_str.split(',')]
                if len(coords) == 2:
                    return tuple(coords)
                else:
                    logger.warning(f"Answer string does not contain exactly 2 coordinates: {answer}")
                    return None
            else:
                logger.warning(f"Answer format not supported: {type(answer)} - {answer}")
                return None
        except Exception as e:
            # logger.warning(f"Error parsing answer: {e}")
            return None

    # TODO:: complete these methods 
    def temperature_cycling(self):
        ...
    
    def persona_based_prompting(self):
        ...