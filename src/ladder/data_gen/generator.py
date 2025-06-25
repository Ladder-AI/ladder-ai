from ladder.data_gen.schema import Transformation, Problem, Dataset, Example, SubProblem
from ladder.engines import VerificationEngine, DifficultyEngine, LLMEngine
from ladder.data_gen.steps import (
    ProblemGenerator, 
    ProblemVerification, 
    TransformationLibrary, 
    VariantGenerator, 
    RecursiveVariantsTree
)
from typing_extensions import Optional, Literal
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
    answer_export_format: Literal["str", "dict", "list", "tuple", "set", "int"] = "str"


class DatasetGenerator:
    """Generate datasets for specific problems with clean, decoupled components"""
    
    def __init__(
        self,
        llm_engine: LLMEngine,
        verification_engine: VerificationEngine,
        small_llm_engine: Optional[LLMEngine] = None,
        difficulty_engine: Optional[DifficultyEngine] = None,
        config: Optional[GenerationConfig] = None
    ):
        self.llm_engine = llm_engine
        self.verification_engine = verification_engine
        self.difficulty_engine = difficulty_engine
        self.config = config or GenerationConfig()
        
        # Initialize LLM tester with small LLM or fallback to main LLM
        test_llm = small_llm_engine or llm_engine
        self.llm_tester = LLMTester(test_llm, verification_engine, difficulty_engine, answer_export_format=config.answer_export_format)
        self.subproblem_generator = SubProblemGenerator(self.llm_tester, self.config)
        
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
        initial_problems: Optional[list[Problem]] = None,
        max_dataset_size: Optional[int] = None,
        examples: Optional[list[Example]] = None,
        auto_generate_solutions: bool = False
    ) -> Optional[Dataset]:
        """Main entry point for dataset generation"""
        initial_problems = initial_problems or []
        max_dataset_size = max_dataset_size or self.config.max_dataset_to_generate
        examples = examples or []
        
        with dspy.settings.context(lm=self.llm_engine.lm, show_guidelines=False):
            # Verify problem suitability
            if not self._verify_problem_suitability(problem_description):
                return None
                
            # Initialize dataset with verified problems
            problems, intelligence_ratio, used_transformations = self._initialize_dataset(
                problem_description, initial_problems, max_dataset_size,
                examples, auto_generate_solutions
            )
            
            # Generate transformations
            transformations = self._generate_transformations(
                problem_description, intelligence_ratio, used_transformations
            )
            
            # Generate variants
            problems = self._generate_variants(problems, transformations)
            
            # Generate recursive subproblems
            self._generate_recursive_subproblems(problems)
            
        return Dataset(
            description=problem_description,
            problems=problems,
            model_intelligence_ratio=intelligence_ratio
        )
    
    def _verify_problem_suitability(self, problem_description: str) -> bool:
        """Verify if the problem is suitable for Ladder algorithm"""
        logger.debug("Verifying problem suitability for Ladder algorithm")
        
        result = self.problem_verifier(problem_description=problem_description)

        if not result.is_ladder_suitable:
            logger.error(f"Problem not suitable: {result.suitability_analysis}")
            return False
            
        logger.success(f"Problem verified successfully")
        return True
    
    def _generate_single_problem(
        self, 
        problem_description: str, 
        examples: list[Example],
        auto_generate_solutions: bool = False
    ) -> Problem:
        """Generate a single new problem"""
        problem: Problem = self.problem_generator(
            problem_description=problem_description, 
            examples=examples
        ).new_problem
        
        # Auto-generate solution using verification engine if requested
        if auto_generate_solutions and self.verification_engine:
            correct_answer = self.verification_engine.get_correct_answer(problem.question)
            if correct_answer:
                problem.correct_answer = correct_answer
                
        return problem
    
    def _initialize_dataset(
        self,
        problem_description: str,
        initial_problems: list[Problem],
        target_size: int,
        examples: list[Example],
        auto_generate_solutions: bool
    ) -> tuple[list[Problem], float, list[Transformation]]:
        """Initialize dataset with verified problems"""
        logger.debug("Initializing dataset")
        
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
                weighted_success_sum, difficulty_sum, examples, auto_generate_solutions
            )
            problems.extend(new_problems)
            weighted_success_sum, difficulty_sum = new_stats
            transformations_used.extend(new_transforms)
        
        intelligence_ratio = (weighted_success_sum / difficulty_sum) if difficulty_sum > 0 else 0.0
        intelligence_ratio = min(intelligence_ratio, 1.0)
        
        logger.success(f"Dataset initialized with {len(problems)} problems")
        return problems, intelligence_ratio, transformations_used
    
    def _generate_additional_problems(
        self,
        problem_description: str,
        count: int,
        initial_weighted_sum: float,
        initial_difficulty_sum: float,
        examples: list[Example],
        auto_generate_solutions: bool
    ) -> tuple[list[Problem], tuple[float, float], list[Transformation]]:
        """Generate additional problems to reach target dataset size"""
        problems = []
        transformations_used = []
        weighted_success_sum = initial_weighted_sum
        difficulty_sum = initial_difficulty_sum
        
        make_easier = None
        trials_count = 0
        current_problem = None
        
        while len(problems) < count:
            # Generate new problem or reset trials
            if make_easier is None or trials_count >= self.config.max_trials_per_problem:
                current_problem = self._generate_single_problem(
                    problem_description, examples, auto_generate_solutions
                )
                transformations = []
                trials_count = 0
            
            # Adjust problem difficulty if needed
            elif make_easier is not None:
                logger.info(f"Making problem {'easier' if make_easier else 'harder'}")
                current_intelligence = (weighted_success_sum / difficulty_sum 
                                     if difficulty_sum > 0 else 0.0)
                current_problem, transformations = self.llm_tester.adjust_problem_difficulty(
                    problem=current_problem,
                    make_easier=make_easier,
                    intelligence_ratio=min(current_intelligence, 1.0)
                )
                transformations_used.extend(transformations)
            
            # Test problem with small LLM - now returns both ratio and answer
            verification_ratio = self.llm_tester.test_problem_solvability(current_problem)
            logger.debug(f"Verification ratio: {verification_ratio}")
            
            if verification_ratio < self.config.unsolved_threshold:
                logger.info("Problem too hard for small LLM - adding to dataset")
                problems.append(current_problem)
                make_easier = True
                trials_count = 0
            elif verification_ratio > self.config.partial_solved_threshold:
                logger.info("Problem too easy for small LLM - making harder")
                current_problem.is_solvable = True
                make_easier = False
                trials_count += 1
                
                if trials_count >= self.config.max_trials_per_problem:
                    problems.append(current_problem)
                    trials_count = 0
            else:
                logger.info("Problem difficulty optimal - adding to dataset")
                current_problem.is_solvable = True
                problems.append(current_problem)
                make_easier = None
                trials_count = 0
            
            # Update statistics
            weighted_success_sum += self._estimate_intelligence_level(
                verification_ratio, current_problem.difficulty_level
            )
            difficulty_sum += current_problem.difficulty_level
        
        return problems, (weighted_success_sum, difficulty_sum), transformations_used
    
    def _generate_transformations(
        self,
        problem_description: str,
        intelligence_ratio: float,
        used_transformations: list[Transformation]
    ) -> list[Transformation]:
        """Generate transformation library for the problem domain"""
        logger.debug("Generating transformations")
        
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
        easier_transforms = self._parse_transformations(easier_result.transformations or [])
        harder_transforms = self._parse_transformations(harder_result.transformations or [])
        
        all_transformations = easier_transforms + harder_transforms + used_transformations
        random.shuffle(all_transformations)
        
        logger.success(f"Generated {len(all_transformations)} transformations")
        return all_transformations
    
    def _generate_variants(
        self,
        problems: list[Problem],
        transformations: list[Transformation]
    ) -> list[Problem]:
        """Generate problem variants for data augmentation"""
        logger.debug("Generating problem variants")

        # No need to test the solvability of the variance as we gonna do this next step while generating subproblems

        
        all_problems = problems.copy()
        
        for problem in problems:
            selected_transforms = random.choices(
                transformations, 
                k=min(self.config.max_variants_per_problem, len(transformations))
            )
            
            variants: list[Problem] = self.variant_generator(
                transformations=selected_transforms,
                problem=problem
            ).variants

            # Test each variant with small LLM to get their answers
            for variant in variants:
                self.llm_tester.test_problem_solvability(variant)

            all_problems.extend(variants)
        
        logger.success(f"Generated {len(all_problems) - len(problems)} variants")
        return all_problems
    
    def _generate_recursive_subproblems(self, problems: list[Problem]) -> None:
        """Generate recursive subproblems for each problem"""
        logger.debug("Generating recursive subproblems")
        
        for problem in problems:
            n_subproblems = random.randint(
                self.config.min_subproblems,
                self.config.max_subproblems
            )
            
            subproblems = self.subproblem_generator.generate_optimized_subproblems(
                base_problem=problem,
                n=n_subproblems
            )
            
            problem.sub_problems = subproblems
            logger.debug(f"Generated {len(subproblems)} optimized subproblems")
    
    # Helper methods
    def _estimate_intelligence_level(self, success_ratio: float, difficulty: float) -> float:
        """Estimate intelligence level based on success ratio and difficulty"""
        return success_ratio * difficulty
    
    def _process_initial_problems(
        self, 
        initial_problems: list[Problem]
    ) -> tuple[list[Problem], tuple[float, float]]:
        """Process and verify initial problems"""
        valid_problems = []
        weighted_success_sum = 0.0
        difficulty_sum = 0.0
        partial_solved_count = 0
        
        for problem in initial_problems:
            verification_ratio = self.llm_tester.test_problem_solvability(problem)
            
            if verification_ratio < self.config.unsolved_threshold:
                valid_problems.append(problem)
                logger.debug("Problem suitable - too hard for small LLM")
            else:
                problem.is_solvable = True
                if (verification_ratio < self.config.partial_solved_threshold and 
                    partial_solved_count < 2):
                    valid_problems.append(problem)
                    partial_solved_count += 1
                logger.debug("Problem partially solvable by small LLM")
            
            weighted_success_sum += self._estimate_intelligence_level(
                verification_ratio, problem.difficulty_level
            )
            difficulty_sum += problem.difficulty_level
        
        return valid_problems, (weighted_success_sum, difficulty_sum)
    
    def _parse_transformations(self, raw_transformations: list[str]) -> list[Transformation]:
        """Parse transformation strings into Transformation objects"""
        if not raw_transformations: 
            return []
            
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
    small_llm_engine: Optional[LLMEngine] = None,
    difficulty_engine: Optional[DifficultyEngine] = None,
    **config_kwargs 
) -> DatasetGenerator:
    """Factory function to create a DatasetGenerator with custom configuration"""
    # config = GenerationConfig(**config_kwargs) if config_kwargs else GenerationConfig()
    config = GenerationConfig()

    # Override default configs 
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return DatasetGenerator(
        llm_engine=llm_engine,
        verification_engine=verification_engine, 
        small_llm_engine=small_llm_engine,
        difficulty_engine=difficulty_engine,
        config=config
    )


class LLMTester:
    """Clean interface for testing Small LLM (to be tuned) capabilities on problems and subproblems 
    
    (used during any new generated problem)"""
    
    def __init__(
        self, 
        llm_engine: LLMEngine,
        verification_engine: VerificationEngine,
        difficulty_engine: Optional[DifficultyEngine] = None,
        answer_export_format: str = "str"
    ):
        self.small_llm_engine = llm_engine
        self.verification_engine = verification_engine
        self.difficulty_engine = difficulty_engine
        self.answer_export_format = answer_export_format
    
    def test_problem_solvability(self, problem: Problem) -> float:
        """Test if LLM can solve a problem and return verification ratio"""
        llm_answer = self.small_llm_engine.generate(prompt=problem.question, exported_format=self.answer_export_format)
        
        # Store the small LLM answer in the problem
        problem.small_llm_answer = llm_answer

        verification_ratio = self.verification_engine.verify(
            problem_question=problem.question, 
            given_answer=llm_answer
        )
        
        return verification_ratio
    
    def test_subproblem_solvability(self, subproblem: SubProblem) -> float:
        """Test if LLM can solve a subproblem and return verification ratio"""
        llm_answer = self.small_llm_engine.generate(prompt=subproblem.question, exported_format=self.answer_export_format)
        logger.success(f"llm_answer: {llm_answer}")
        
        # Store the small LLM answer in the subproblem (if SubProblem has small_llm_answer field)
        if hasattr(subproblem, 'small_llm_answer'):
            subproblem.small_llm_answer = llm_answer
        
        verification_ratio = self.verification_engine.verify(
            problem_question=subproblem.question, 
            given_answer=llm_answer
        )
        
        return verification_ratio
    
    def adjust_problem_difficulty(
        self, 
        problem: Problem, 
        make_easier: bool, 
        intelligence_ratio: float = 0.8
    ) -> tuple[Problem, list[Transformation]]:
        """Adjust problem difficulty based on test results"""
        if not self.difficulty_engine:
            return problem, []
        
        return self.difficulty_engine.change_problem_difficulty(
            problem=problem,
            model_intelligence_ratio=intelligence_ratio,
            increase_difficulty=not make_easier
        )
    
    def adjust_subproblem_difficulty(
        self, 
        subproblem: SubProblem, 
        make_easier: bool, 
        intelligence_ratio: float = 0.8
    ) -> tuple[SubProblem, list[Transformation]]:
        """Adjust subproblem difficulty based on test results"""
        if not self.difficulty_engine:
            return subproblem, []
        
        return self.difficulty_engine.change_subproblem_difficulty(
            subproblem=subproblem,
            model_intelligence_ratio=intelligence_ratio,
            increase_difficulty=not make_easier
        )


class SubProblemGenerator:
    """Generate and optimize subproblems for main problems (used in variants generation step)
    
    (used during last step of recursive Tree generation process)
    """
    
    def __init__(self, llm_tester: LLMTester, config: GenerationConfig):
        self.llm_tester = llm_tester
        self.config = config
    
    def generate_optimized_subproblems(self, base_problem: Problem, n: int = 3) -> list[SubProblem]:
        """Generate subproblems optimized for the target LLM's capability"""
        
        # 1-  test the smaller LLM on the main problem to get baseline performance
        current_verification_ratio = self.llm_tester.test_problem_solvability(base_problem)
        
        subproblems = []

        # Start with the same problem but as a SubProblem instance
        small_llm_answer = self.llm_tester.small_llm_engine.generate(prompt=base_problem.question, 
                                                                     exported_format=self.llm_tester.answer_export_format)
        current_subproblem = SubProblem(
            question=base_problem.question,
            correct_answer=base_problem.correct_answer,
            difficulty_level=base_problem.difficulty_level,
            is_solvable=base_problem.is_solvable,
            small_llm_answer=small_llm_answer
        )
        max_iterations = 3

        for iteration in range(max_iterations):
            logger.debug(f"Iteration {iteration + 1}: Verification ratio = {current_verification_ratio}")
            
            if current_verification_ratio < self.config.unsolved_threshold:
                # Too hard - make easier
                logger.debug(f"Subproblem too hard (ratio: {current_verification_ratio}), making easier")
                current_subproblem, _ = self.llm_tester.adjust_subproblem_difficulty(
                    current_subproblem, make_easier=True
                )
                # Test the adjusted subproblem with smaller LLM
                subproblems.append(current_subproblem)
                current_verification_ratio = self.llm_tester.test_subproblem_solvability(current_subproblem)
                
            elif current_verification_ratio > self.config.partial_solved_threshold:
                # Too easy - make harder
                logger.debug(f"Subproblem too easy (ratio: {current_verification_ratio}), making harder")
                # No need to add it as the problem is too easy

                current_subproblem, _ = self.llm_tester.adjust_subproblem_difficulty(
                    current_subproblem, make_easier=False
                )
                # Test the adjusted subproblem with smaller LLM
                current_verification_ratio = self.llm_tester.test_subproblem_solvability(current_subproblem)
                
            else:
                # Difficulty is optimal - break out of loop
                logger.debug(f"Subproblem difficulty optimal (ratio: {current_verification_ratio})")
                subproblems.append(current_subproblem)
                break
        
        
        return subproblems
    
    def _optimize_subproblem_difficulty(
        self, 
        subproblem: SubProblem, 
        base_verification_ratio: float
    ) -> SubProblem:
        """Optimize subproblem difficulty through iterative testing"""
        current_subproblem = subproblem
        
   
        # Ensure the final subproblem has been tested and has small_llm_answer
        if not hasattr(current_subproblem, 'small_llm_answer') or current_subproblem.small_llm_answer is None:
            logger.debug("Final test to ensure small_llm_answer is captured")
            self.llm_tester.test_subproblem_solvability(current_subproblem)
        
        return current_subproblem 


