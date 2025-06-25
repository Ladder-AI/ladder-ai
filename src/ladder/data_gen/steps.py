
"""
List of Steps to generate Dataset for Ladder Finetuning
"""
from ladder.data_gen.schema import SubProblem, Problem, Example
from typing_extensions import  Optional
import dspy 



class ProblemVerification(dspy.Signature):
    """
    You are a problem generation expert specializing in dataset creation for machine learning models.

    Assess whether the given problem is suitable for Ladder Finetuning, a methodology that relies on structured difficulty decomposition.

    Evaluate the problem based on:
    1. **Automation** – Can the solution be automatically verified (no human evaluation required)?
    2. **Scalability** – Can the problem be expressed at multiple difficulty levels?
    3. **Variability** – Can diverse variants be generated while preserving its core?

    Provide a clear analysis and a final suitability decision.
    """

    problem_description: str = dspy.InputField(
        prefix="Problem Description: ",
        format=str,
        desc="Full description of the problem including context, constraints, and what constitutes a correct solution"
    )

    suitability_analysis: str = dspy.OutputField(
        format=str,
        desc="Detailed justification of the suitability assessment, addressing automation, scalability, and variability"
    )

    is_ladder_suitable: bool = dspy.OutputField(
        format=bool,
        desc="True if the problem is suitable for Ladder Finetuning; False otherwise"
    )

class ProblemGenerator(dspy.Signature):
    """
    You are a problem generation expert specializing in dataset creation.

    Generate a new, complete problem instance from the provided Problem description.

    Optionally, you may be provided with example problems and their solutions
    to help guide the generation process.

    Ensure the problem is well-formed, diverse, and suitable for dataset inclusion. and your generated problem is well matched with what the 
    user gives you in the problem description 
    """

    problem_description: str = dspy.InputField(
        prefix="Problem Description: ",
        format=str,
        desc="General description of the problem type, its constraints, and solution expectations"
    )

    examples: Optional[list[Example]] = dspy.InputField(
        prefix="Examples (optional):\n",
        format=list[Example],
        desc="List of example problems and their corresponding solutions as question and answer format",
    )

    new_problem: "Problem" = dspy.OutputField(
        format="Problem",
        desc="Generated problem instance with question, solution, and difficulty level (float between 0.0 and 1.0)"
    )

class TransformationLibrary(dspy.Signature):
    """
    You are a problem generation expert specializing in dataset creation for machine learning models.

    Generate a library of problem transformations that can increase or decrease difficulty. These transformations will be used to create controlled variants of a problem for Ladder Finetuning.

    Each transformation should:
    - Be precise and actionable
    - Maintain the problem’s integrity and solvability
    - Include a difficulty impact score (0.0–1.0)

    Examples: Add constraints, simplify inputs, require intermediate steps, introduce noise, etc.
    Examples for math could be increase the equation degrees, complexity or make it simpler, .. .
    """

    problem_description: str = dspy.InputField(
        prefix="Problem Domain: ",
        format=str,
        desc="Description of the problem context for which transformations should be designed"
    )

    model_intelligence_ratio: float = dspy.InputField(
        prefix="Model Capability Threshold: ",
        format=float,
        desc="Difficulty threshold (0.0–1.0); transformations above this make the problem harder, below make it easier"
    )

    make_easier: bool = dspy.InputField(
        prefix="Include Easier Variants: ",
        format=bool,
        desc="True to include transformations that reduce problem difficulty; False to focus on increasing/maintaining difficulty"
    )

    transformations: list[str] = dspy.OutputField(
        format=list[str],
        desc="List of transformation rules, each formatted as '<description> || <difficulty_impact>' (e.g., 'Add distractors to answer options || 0.7')"
    )

class VariantGenerator(dspy.Signature):
    """
    You are a problem generation expert specializing in dataset creation.

    Generate multiple variants of a base problem by applying provided transformations. Each variant should:
    - Preserve the core intent of the original problem
    - Introduce meaningful variation
    - Make sure all generated problems have variety and different structures 

    Note: the given transformations are helpful but you can adjust the problem as needed

    Apply transformations strategically—not randomly—to maximize dataset diversity and utility.
    """

    transformations: list[str] = dspy.InputField(
        prefix="Available Transformations: ",
        format=list[str],
        desc="List of transformation rules formatted as '<transformation_description> || <difficulty_level>'"
    )

    problem: Problem = dspy.InputField(
        prefix="Base Problem: ",
        format=Problem,
        desc="Original problem instance to transform into multiple variants"
    )

    variants: list[Problem] = dspy.OutputField(
        format=list[Problem],
        desc="List of new problem instances created from the base problem using the provided transformations"
    )

class RecursiveVariantsTree(dspy.Signature):
    """
    You are a problem generation expert specializing in dataset creation for machine learning models.

    Generate a hierarchical tree of progressively easier problem variants starting from a root problem. 
    This structure supports Ladder Finetuning by gradually reducing difficulty across multiple levels.

    Each variant must:
    - Be simpler than its parent
    - Maintain semantic and logical coherence
    - Align with model learning progression
    """

    transformations: list[str] = dspy.InputField(
        prefix="Available Transformations: ",
        format=list[str],
        desc="List of transformations that reduce difficulty, formatted as '<description> || <difficulty_impact>'"
    )

    problem: Problem = dspy.InputField(
        prefix="Root Problem: ",
        format=Problem,
        desc="Most challenging version of the problem from which easier variants will be recursively derived"
    )

    n: int = dspy.InputField(
        prefix="Number of Difficulty Levels: ",
        format=int,
        desc="Number of recursive steps to take, each producing an easier variant"
    )

    sub_variants: list[SubProblem] = dspy.OutputField(
        format=list[SubProblem],
        desc="List of problem variants arranged from hardest (root) to easiest, forming a difficulty tree"
    )
