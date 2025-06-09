from ladder.data_gen.schema import Transformation, Problem, Dataset
from ladder.engines import VerificationEngine, DifficultyEngine, LLMEngine
from ladder.data_gen.steps import (
    ProblemGenerator, 
    ProblemVerification, 
    TransformationLibrary, 
    VariantGenerator, 
    RecursiveVariantsTree,
    _SubProblemTester
)
from typing_extensions import Annotated, Doc
from typing import Optional, Tuple, List
from loguru import logger
import random 
import dspy 
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for dataset generation process"""
    max_variants_per_problem: int = 5
    min_subproblems: int = 3
    max_subproblems: int = 5
    max_trials_per_problem: int = 3
    partial_solved_threshold: float = 0.8
    unsolved_threshold: float = 0.5
    max_dataset_to_generate: int = 10

# TODO:: add customize function to adjust configurations 

class DatasetGenerator:
    """Generate datasets for specific problems with decoupled components.
    
    The generator is now independent of problem descriptions and initial problems,
    making it more flexible and reusable.
    """
    
    def __init__(
        self,
        llm_engine: Annotated[LLMEngine, Doc("LLM Engine for dataset generation")],
        verification_engine: Annotated[VerificationEngine, Doc("Problem verification engine")],
        difficulty_engine: Annotated[Optional[DifficultyEngine], Doc("Difficulty engine")] = None,
        config: Optional[GenerationConfig] = None
    ):
        self.llm_engine = llm_engine
        self.verification_engine = verification_engine
        self.difficulty_engine = difficulty_engine
        self.config = config or GenerationConfig()
        
        # Initialize DSPy modules
        self._setup_modules()
        
    def _setup_modules(self):
        """Initialize all DSPy modules"""
        self.problem_verifier = dspy.ChainOfThought(ProblemVerification)
        self.problem_generator = dspy.ChainOfThought(ProblemGenerator)
        self.transformation_generator = dspy.ChainOfThought(TransformationLibrary)
        self.variant_generator = dspy.ChainOfThought(VariantGenerator)
        self.recursive_tree_generator = dspy.ChainOfThought(RecursiveVariantsTree)
        
    def generate_dataset(
        self,
        problem_description: str,
        initial_problems: Optional[List[Problem]] = None,
        max_dataset_size: Optional[int] = None
    ) -> Optional[Dataset]:
        """Main entry point for dataset generation.
        
        Args:
            problem_description: Description of the problem domain
            initial_problems: Starting set of problems (optional)
            max_dataset_size: Maximum number of problems to generate
            
        Returns:
            Generated dataset or None if verification fails
        """
        initial_problems = initial_problems or []
        max_dataset_size = max_dataset_size or self.config.max_dataset_to_generate
        
        with dspy.settings.context(lm=self.llm_engine.lm, show_guidelines=False):
            # Verify problem suitability
            if not self.verify_problem_suitability(problem_description):
                return None
                
            # Initialize dataset
            problems, intelligence_ratio, used_transformations = self.initialize_dataset(
                problem_description, initial_problems, max_dataset_size
            )
            
            # Generate transformations
            transformations = self.generate_transformations(
                problem_description, intelligence_ratio, used_transformations
            )
            
            # Generate variants
            problems = self.generate_variants(problems, transformations)
            
            # Generate recursive variants
            self.generate_recursive_variants(problems, transformations)
            
        return Dataset(
            description=problem_description,
            problems=problems,
            model_intelligence_ratio=intelligence_ratio
        )
    
    def verify_problem_suitability(self, problem_description: str) -> bool:
        """Verify if the problem is suitable for Ladder algorithm."""
        logger.info("Verifying problem suitability for Ladder algorithm")
        
        result = self.problem_verifier(problem_description=problem_description)
        
        if not result.is_ladder_suitable:
            logger.error(f"Problem not suitable: {result.description}")
            return False
            
        logger.success(f"Problem verified: {result.description}")
        return True
    
    def initialize_dataset(
        self,
        problem_description: str,
        initial_problems: List[Problem],
        target_size: int
    ) -> Tuple[List[Problem], float, List[Transformation]]:
        """Initialize dataset with verified problems."""
        logger.info("Initializing dataset")
        
        problems = []
        transformations_used = []
        weighted_success_sum = 0.0
        difficulty_sum = 0.0
        
        # Process initial problems
        problems, stats = self._process_initial_problems(initial_problems)
        weighted_success_sum, difficulty_sum = stats
        
        # Generate additional problems if needed
        if len(problems) < target_size:
            new_problems, new_stats, new_transforms = self._generate_additional_problems(
                problem_description, target_size - len(problems), 
                weighted_success_sum, difficulty_sum
            )
            problems.extend(new_problems)
            weighted_success_sum, difficulty_sum = new_stats
            transformations_used.extend(new_transforms)
        
        intelligence_ratio = (weighted_success_sum / difficulty_sum) if difficulty_sum > 0 else 0.0
        intelligence_ratio = min(intelligence_ratio, 1.0)
        
        logger.success(f"Dataset initialized with {len(problems)} problems")
        return problems, intelligence_ratio, transformations_used
    
    def generate_transformations(
        self,
        problem_description: str,
        intelligence_ratio: float = 0.0,
        used_transformations: List[Transformation] = []
    ) -> List[Transformation]:
        """Generate transformation library for the problem domain."""
        logger.info("Generating transformations")
        
        # Generate easier transformations
        easier_result = self.transformation_generator(
            problem_description=problem_description,
            model_intelligence_ratio=intelligence_ratio,
            make_easier=True
        )
        
        # Generate harder transformations
        harder_result = self.transformation_generator(
            problem_description=problem_description,
            model_intelligence_ratio=intelligence_ratio,
            make_easier=False
        )
        
        # Parse and combine transformations
        easier_transforms = self._parse_transformations(easier_result.transformations)
        harder_transforms = self._parse_transformations(harder_result.transformations)
        
        all_transformations = easier_transforms + harder_transforms + used_transformations
        random.shuffle(all_transformations)
        
        logger.success(f"Generated {len(all_transformations)} transformations")
        return all_transformations
    
    def generate_variants(
        self,
        problems: List[Problem],
        transformations: List[Transformation]
    ) -> List[Problem]:
        """Generate problem variants for data augmentation."""
        logger.info("Generating problem variants")
        
        all_problems = problems.copy()
        
        for problem in problems:
            selected_transforms = random.choices(
                transformations, 
                k=min(self.config.max_variants_per_problem, len(transformations))
            )
            
            result = self.variant_generator(
                transformations=selected_transforms,
                problem=problem
            )
            
            all_problems.extend(result.variants)
        
        logger.success(f"Generated {len(all_problems) - len(problems)} variants")
        return all_problems
    
    def generate_recursive_variants(
        self,
        problems: List[Problem],
        transformations: List[Transformation]
    ) -> None:
        """Generate recursive subproblems for each problem."""
        logger.info("Generating recursive variants")
        
        subproblem_tester = _SubProblemTester(
            llm_engine=self.llm_engine,
            difficulty_engine=self.difficulty_engine
        )
        
        for problem in problems:
            n_subproblems = random.randint(
                self.config.min_subproblems,
                self.config.max_subproblems
            )
            
            subproblems = subproblem_tester.generate_subproblems(
                base_problem=problem,
                n=n_subproblems
            )
            
            problem.sub_problems = subproblems
            logger.debug(f"Generated {len(subproblems)} subproblems")
    
    def generate_single_problem(self, problem_description: str) -> Problem:
        """Generate a single new problem."""
        return self.problem_generator(problem_description=problem_description).new_problem
    
    def estimate_intelligence_level(self, success_ratio: float, difficulty: float) -> float:
        """Estimate intelligence level based on success ratio and difficulty."""
        return success_ratio * difficulty
    
    # Private helper methods
    def _process_initial_problems(
        self, 
        initial_problems: List[Problem]
    ) -> Tuple[List[Problem], Tuple[float, float]]:
        """Process and verify initial problems."""
        valid_problems = []
        weighted_success_sum = 0.0
        difficulty_sum = 0.0
        partial_solved_count = 0
        
        for problem in initial_problems:
            verification_ratio = self.verification_engine.verify(problem=problem)
            
            if verification_ratio < self.config.unsolved_threshold:
                valid_problems.append(problem)
            else:
                problem.is_solvable = True
                if (verification_ratio < self.config.partial_solved_threshold and 
                    partial_solved_count < 2):
                    valid_problems.append(problem)
                    partial_solved_count += 1
            
            weighted_success_sum += self.estimate_intelligence_level(
                verification_ratio, problem.difficulty_level
            )
            difficulty_sum += problem.difficulty_level
        
        return valid_problems, (weighted_success_sum, difficulty_sum)
    
    def _generate_additional_problems(
        self,
        problem_description: str,
        count: int,
        initial_weighted_sum: float,
        initial_difficulty_sum: float
    ) -> Tuple[List[Problem], Tuple[float, float], List[Transformation]]:
        """Generate additional problems to reach target dataset size."""
        problems = []
        transformations_used = []
        weighted_success_sum = initial_weighted_sum
        difficulty_sum = initial_difficulty_sum
        
        make_easier = None
        trials_count = 0
        current_problem = None
        
        while len(problems) < count:
            # Generate or modify problem
            if make_easier is None or trials_count >= self.config.max_trials_per_problem:
                current_problem = self.generate_single_problem(problem_description)
                transformations = []
                trials_count = 0
            elif self.difficulty_engine:
                current_intelligence = (weighted_success_sum / difficulty_sum 
                                     if difficulty_sum > 0 else 0.0)
                current_problem, transformations = self.difficulty_engine.change_problem_difficulty(
                    problem=current_problem,
                    model_intelligence_ratio=min(current_intelligence, 1.0),
                    increase_difficulty=not make_easier
                )
                transformations_used.extend(transformations)
            
            # Verify problem
            verification_ratio = self.verification_engine.verify(problem=current_problem)
            
            if verification_ratio < self.config.unsolved_threshold:
                problems.append(current_problem)
                make_easier = True
            else:
                current_problem.is_solvable = True
                make_easier = False
                trials_count += 1
                
                if verification_ratio < self.config.partial_solved_threshold:
                    problems.append(current_problem)
            
            # Update statistics
            weighted_success_sum += self.estimate_intelligence_level(
                verification_ratio, current_problem.difficulty_level
            )
            difficulty_sum += current_problem.difficulty_level
        
        return problems, (weighted_success_sum, difficulty_sum), transformations_used
    
    def _parse_transformations(self, raw_transformations: List[str]) -> List[Transformation]:
        """Parse transformation strings into Transformation objects."""
        parsed = []
        for item in raw_transformations:
            if "||" not in item:
                continue
            try:
                description, difficulty = map(str.strip, item.split("||"))
                parsed.append(Transformation(
                    description=description,
                    difficulty_level=float(difficulty)
                ))
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse transformation: {item}")
                continue
        return parsed


def create_dataset_generator(
    llm_engine: LLMEngine,
    verification_engine: VerificationEngine,
    difficulty_engine: Optional[DifficultyEngine] = None,
    **config_kwargs
) -> DatasetGenerator:
    """Factory function to create a DatasetGenerator with custom configuration."""
    config = GenerationConfig(**config_kwargs) if config_kwargs else GenerationConfig()
    return DatasetGenerator(llm_engine, verification_engine, difficulty_engine, config)