from typing_extensions import  Optional, Any, Callable, Union
from ladder.utils import generate_hf_model_answer
from ladder.finetuning.grpo import grpo
from ladder.schema import  Problem, problems_to_vladder
from ladder.engines import VerificationEngine
from ladder.data_gen.generator import DatasetGenerator, create_dataset_generator
from ladder.config import LadderConfig
from ladder import setup_default_engines
from loguru import logger
import sys 

try:
    from transformers import PreTrainedModel
except ImportError:
    logger.error("transformers not installed. Please install with `pip install transformers`")
    sys.exit(1)
    # raise


# Note: TTRL:: is not an exact finetuning algorithm through which u will update the base model once and use it 
# every where but instead u just tune it in the test time 

class TTRL:
    """Finetuning Engine using the TTRL Algorithm."""

    def __init__(
        self,
        *,
        base_model: PreTrainedModel,
        config: Optional[LadderConfig] = None,
        verification_engine: Optional[VerificationEngine] = None,
        dataset_generator: Optional[DatasetGenerator] = None,
        reward_funcs: list[Callable] = [],
    ):
        self.config = config
        self.base_model = base_model
        self.reward_funcs = reward_funcs

        # Ensure dataset_generator is available
        if dataset_generator is None:
            if not config:
                logger.error("'config' must be provided if 'dataset_generator' is not provided.")
                sys.exit(1)

            llm_engine, default_verification_engine, difficulty_engine = setup_default_engines(config)
            verification_engine = verification_engine or default_verification_engine
            dataset_generator = create_dataset_generator(
                llm_engine=llm_engine,
                verification_engine=verification_engine,
                difficulty_engine=difficulty_engine,
            )

        # Ensure verification_engine is available
        if verification_engine is None:
            if not config:
                logger.error("'config' must be provided if 'verification_engine' is not provided.")
                sys.exit(1)
            _, verification_engine, _ = setup_default_engines(config)

        self.dataset_generator = dataset_generator
        self.verification_engine = verification_engine


        logger.debug("Generating transformations...")
        self.transformations = self._generate_transformations()

    
    def _generate_transformations(self, problem_description: str):
        """ prepare list of transformations to be applied to the tuned model during ttrl """
        transformations = self.dataset_generator.generate_transformations(
            problem_description=problem_description,
            intelligence_ratio = 0.5 # TODO:: could be loaded from old tuned model 
        )
        return transformations

    
    def generate(self,prompt:Union[str,Problem],nvariants:int=3, *args, **kwargs) -> str:
        """
            Implements TTRL finetuning as per Algorithm 2 in the LADDER paper.
            Qtest: List of test integrals
            N: Number of variants to generate per test integral

            Args:
            prompt: str >> question for the model
        """
        if not self.transformations:
            self.transformations = self._generate_transformations()
        
        problem = None
        if isinstance(prompt, Problem):
            problem = prompt 
        elif isinstance(prompt, str):
            problem = self.dataset_generator.generate_single_problem(prompt)
        else:
            raise ValueError("Invalid prompt type. Expected str or Problem.")

        # 1- generate N variants for each question in Q_test # TODO:: make sure it is vladder compatiable

        v_ttrl = self.dataset_generator.generate_variants(
                                            problems=[problem], 
                                            transformations=self.transformations
                                            ) 

        # 2- apply GRPO over the base (tuned model) and variants
        v_ttrl = vttrl_to_vladder(v_ttrl) # make it ladder schema compatible

        # 3- apply grpo over the tuned model using the generated variants 
        model = grpo(v_ttrl, self.base_model, config=self.config, reward_funcs=self.reward_funcs)

        # 4- test the tuned model and return answer for the question
        answer = generate_hf_model_answer(model,prompt)

        return answer
    


def vttrl_to_vladder(vttrl: list[Problem]):
    return problems_to_vladder(vttrl)