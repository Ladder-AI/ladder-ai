from ladder.data_gen.schema import  Transformation, Problem, Dataset
from ladder.engines import VerificationEngine, DifficultyEngine
from ladder.data_gen.steps import (ProblemGenerator, 
                                   ProblemVerification, 
                                   TransformationLibrary, 
                                   VariantGenerator, 
                                   RecursiveVariantsTree,
                                   _SubProblemTester)
from typing_extensions import Annotated, Doc
from ladder.engines import LLMEngine
from typing import Optional
from loguru import logger
import random 
import dspy 

# HYPERPARAMETERS
# TODO:: move to configs
INITIAL_DATASET_SIZE = 10 # inital dataset bedore starting in the generation process should be 10


class DatasetGenerator(dspy.Module):
    """ Generate required dataset for specific problem

    1. Transformations
    2. Variants Generation
    3. Recursive Variants Tree 
    """
    
    def __init__(self,*,
                    problem_description: Annotated[str, Doc(
                    """A string containing the problem description, from which the transformations will be defined"""
                    )],
                    llm_engine: Annotated[LLMEngine, Doc(
                    """LLM Engine to be used for dataset generation"""
                    )],
                    verification_engine: Annotated["VerificationEngine", Doc(
                    """Problem Verification Engine"""
                    )],
                    difficulty_engine: Annotated[Optional[DifficultyEngine], Doc(
                    """Difficulty Engine to be used later during step3 to generate N difficulty levels,
                    if none , N will be set to 3"""
                    )] = None,
                    inital_problems: list[Problem] = [],
                    max_dataset_to_generate: Optional[int] = 3
                 ):
        self.problem_description = problem_description
        self.llm_engine = llm_engine
        self.verification_engine = verification_engine
        self.difficulty_engine = difficulty_engine

        self.problems = inital_problems
        self.model_intelligence_ratio = None

        # dataset generation presteps (our new 2 steps added not mentioned in the paper)
        self.problem_verifier = dspy.ChainOfThought(ProblemVerification)
        self._problem_generator = dspy.ChainOfThought(ProblemGenerator)

        # dataset generation steps 
        self.transformations_generation = dspy.ChainOfThought(TransformationLibrary) # step1
        self.variants_generation = dspy.ChainOfThought(VariantGenerator) # step2
        self.recursive_variants_tree_generation = dspy.ChainOfThought(RecursiveVariantsTree) # step3

        self.max_datasets_to_generate = max_dataset_to_generate

    def forward(self) -> Dataset:
        """ Main generation process of dataset done here """

        with dspy.settings.context(lm=self.llm_engine.lm, show_guidelines=False):
            ## Verify Problem for Ladder Algorithm 
            is_verified_problem = self.verify_problem()
            if not is_verified_problem:
                return 
            
            ## initalize problems , test model intelligence 
            inital_problems, model_intelligence_ratio, transformations_used = self.init_dataset()
            logger.success(f"Dataset initialized successfully with {len(inital_problems)} problems")
            self.problems = inital_problems
            self.model_intelligence_ratio = abs(model_intelligence_ratio)

            # # 1- Generate Transformations
            easier_transformations, harder_transformations = self.generate_transformations()
            transformations = easier_transformations + harder_transformations 
            transformations.extend(transformations_used)
            random.shuffle(transformations)
            logger.success(f"{len(transformations)} transformations generated successfully")

            # 2- Generate Variants (other problems and then flatten)
            self.generate_variants(transformations=transformations)

            # 3- Generate Recursive Variants Tree (for each problem try to make it easier)
            self.generate_recursive_variants_tree(transformations=transformations)
            
        dataset = Dataset(
            description=self.problem_description,
            problems=inital_problems,
            model_intelligence_ratio=model_intelligence_ratio
        ) 
        return dataset
        
    def verify_problem(self) -> bool:
        """ PRESTEP1::Verify if the problem is suitable for Ladder Finetuning
        """
        logger.warning("PRE_STEP1:: Verifying Problem for Ladder Algorithm \n")
        out = self.problem_verifier(problem_description=self.problem_description)

        is_ladder_suitable = out.is_ladder_suitable
        description = out.description

        if not is_ladder_suitable:
            logger.error("Problem is not suitable for Ladder Algorithm \n")
            logger.info("Reason: {}".format(description))
            return 
        else:
            logger.success("Problem is suitable for Ladder Algorithm \n")
            logger.info(f"Reason: {description}")
    
        return is_ladder_suitable
    
    def init_dataset(self) -> tuple[list[Problem], float, list[Transformation]]:
        """PRE_STEP2:: Generate initial dataset of problems that will be used in the variants generation process

        it will help in generating inital list of dataset problems if:
        - inital_dataset length is smaller than N (10 ::hyperparameter)
        - the llm to be tuned is able to solve it 

        output from this step
        1. unsolved_problems: list[Problem] = [] # we need to have at least 10 unsolved problems
        2. model_intelligence_ratio / difficulty_threshold 
        3. transformations_used  (those transformations used during the initalization process 
                  which should be useful later in the transformation library to help the llm to define new transformations)

        By the end of this method we have to have at least 10 problems (atleast 80% of them couldbe solved by llm)
        """
        logger.warning("PRE_STEP2:: Generating initial dataset of problems that will be used in the variants generation process \n")

        unsolved_problems: list[Problem] = []
        transformations_used: list[Transformation] = []

        weighted_success_sum = 0.0
        difficulty_sum = 0.0
        _partial_solved_dataset = set()
        j = 0

        logger.info("Filtering inital examples")
        for problem in self.problems:
            verification_ratio = self.verification_engine.verify(problem=problem) # 0: unsolved , 1: solved
            j += 1
            if verification_ratio < 0.5:
                unsolved_problems.append(problem)
            else:
                problem.is_solvable = True
                if verification_ratio < 0.8 and len(_partial_solved_dataset) < 2:
                    _partial_solved_dataset.add(problem)
                    unsolved_problems.append(problem)

            # Accumulate weighted score and difficulty
            weighted_success_sum += self.estimate_current_llm_intelligence_level(
                Si=verification_ratio,
                Di=problem.difficulty_level
            )
            difficulty_sum += problem.difficulty_level

        # 2- complete generating new dataset if < 10
        make_it_easier = None
        max_trials_per_problem = 3

        logger.warning(f"Generating new Datasets")
        current_model_intelligence_ratio = weighted_success_sum / difficulty_sum if weighted_success_sum > 0.0 else 0.0
        current_model_intelligence_ratio = min(current_model_intelligence_ratio, 1.0)
        while len(unsolved_problems) < self.max_datasets_to_generate:
            if make_it_easier is None or max_trials_per_problem >= 3:
                new_problem = self._generate_new_problem()
                transformations = []
                max_trials_per_problem = 0

            elif make_it_easier:
                new_problem, transformations = self.difficulty_engine.change_problem_difficulty(
                    problem=new_problem, 
                    model_intelligence_ratio=current_model_intelligence_ratio,
                    increase_difficulty=False
                )
            else:
                new_problem, transformations = self.difficulty_engine.change_problem_difficulty(
                    problem=new_problem, 
                    model_intelligence_ratio=current_model_intelligence_ratio,
                    increase_difficulty=True
                )
            transformations_used.extend(transformations)

            verification_ratio = self.verification_engine.verify(problem=new_problem)
            j += 1

            if verification_ratio < 0.5: 
                unsolved_problems.append(new_problem)
                make_it_easier = True
            else:
                make_it_easier = False
                new_problem.is_solvable = True
                max_trials_per_problem += 1
                if verification_ratio < 0.8 and len(_partial_solved_dataset) < 2:
                    _partial_solved_dataset.add(new_problem)
                    unsolved_problems.append(new_problem)

            weighted_success_sum += self.estimate_current_llm_intelligence_level(
                Si=verification_ratio,
                Di=new_problem.difficulty_level
            )
            difficulty_sum += new_problem.difficulty_level

            current_model_intelligence_ratio = weighted_success_sum / difficulty_sum if difficulty_sum > 0 else 0
            current_model_intelligence_ratio = min(current_model_intelligence_ratio, 1.0)

        # Compute final intelligence ratio (always in [0, 1])
        model_intelligence_ratio = (weighted_success_sum / difficulty_sum) if difficulty_sum > 0 else 0.0

        logger.success("(PRE_STEP) Generating initial dataset of problems that will be used in the variants generation process is done \n")
        return unsolved_problems, model_intelligence_ratio, transformations_used
            
    def generate_transformations(self) -> tuple[list[Transformation], list[Transformation]]:
        """STEP1:: generate List of transformations that will be used in the variants generation process

            Return:
                easier_transformations: list of transformations to make the problem easier
                harder_transformations: list of transformations to make the problem harder
        
        """
        logger.warning("Step1::  Generating List of Transformations to be used in variant generations \n")

        def _parse_transformations(raw_transformations: list[str]) -> list[Transformation]:
            ## format "description || difficulty_ratio"
            parsed = []
            for item in raw_transformations:
                if "||" not in item:
                    continue  # skip invalid entries
                description, difficulty = map(str.strip, item.split("||"))
                parsed.append(Transformation(description=description, difficulty_level=float(difficulty)))
            return parsed

        unsolvable_problem_example = next((x for x in self.problems if not x.is_solvable), None)
        unsolvable_problem_example = unsolvable_problem_example or self.problems[0] # take any if so 

        easier_transformations = self.transformations_generation(problem_description=self.problem_description,
                                                   model_intelligence_ratio=self.model_intelligence_ratio,
                                                   make_easier=True)

        harder_transformations = self.transformations_generation(problem_description=self.problem_description,
                                                model_intelligence_ratio=self.model_intelligence_ratio,
                                                make_easier=False)
        
        easier_transformations = easier_transformations.transformations
        harder_transformations = harder_transformations.transformations

        # parse Transformations 
        easier_transformations = _parse_transformations(easier_transformations)
        harder_transformations = _parse_transformations(harder_transformations)

        logger.info(f"{len(easier_transformations)} easier transformations have been generated")
        logger.info(f"{len(harder_transformations)} harder transformations have been generated")
        return easier_transformations or [], harder_transformations or []
        
    def generate_variants(self,transformations:list[Transformation]) -> None:
        """STEP2:: generate List of variants that will be used in the variants generation process
            This step works as data augmentation to make the dataset richer and more diverse
        """
        # Loop over each problem we have and generate N variants from each problem 
        # Select List of Random Transformations to be applied to each problem for this variant generation 
        ## TODO:: we should also handle verification process during variant generation to check if the variants are solvable or not
        logger.warning(f"STEP2:: Generate New Variants for each problem")
        
        variants = []
        for problem in self.problems:
            random_transformations = random.choices(transformations, k=5) # TODO:: difficulty engine should do this 
            out = self.variants_generation(transformations=random_transformations, problem=problem)
            variants += out.variants
        self.problems.extend(variants)

    def generate_recursive_variants_tree(self,transformations: list[Transformation]) -> None:
        """STEP3:: generate List of variants that will be used in the variants generation process"""
        logger.warning(f"STEP3:: Generate recursive subproblems per each problem")

        # TODO:: transformations should be selected by difficulty engine
        subproblem_tester = _SubProblemTester(
            llm_engine=self.llm_engine,
            difficulty_engine=self.difficulty_engine
        )
        for problem in self.problems:
            # Generate 3-5 subproblems per each problem
            n = random.randint(3, 5)
            subproblems = subproblem_tester.generate_subproblems(base_problem=problem, n=n)
            # subproblems = self.recursive_variants_tree_generation(problem=problem, 
            #                                                         transformations=random.choices(transformations, k=5),
            #                                                         n=3).sub_variants
            problem.sub_problems = subproblems
            logger.debug(f"Generated {len(subproblems)} subproblems") # subproblems for problem {problem.question}

    def _generate_new_problem(self) -> Problem:
        """ utils to generate new problem"""
        return self._problem_generator(problem_description=self.problem_description).new_problem

    def estimate_current_llm_intelligence_level(self, Si:float, Di:float):
        """ utils to estimate the current intelligence level of the model
            according to 
            - problem_difficulty (Di) where 0 means easy and 1 means super hard
            - solvability_ration (Si) >> where 0 means model failed to solve and 1 means model solved it perfectly

            # (Di * Si - (1 - Di) * (1 - Si) * alpha) , alpha =0.5: boosting factor

            Intelligence Ratio= 

            ∑ Si * Di 
            _____________
            ∑ Di
        """
        # Si * Di - (1 - Di) * (1 - Si) * 0.5 ## ::hyperparameter
        return Si * Di
