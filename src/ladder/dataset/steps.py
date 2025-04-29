from typing_extensions import Doc, Annotated

def generate_transformations(
                    *,
                    problem_description:Annotated[str, Doc(
                    """A string containing the problem description, from which the transformations will be defined"""
                    )],
) -> list[str]:
    """STEP1:: generate List of transformations that will be used in the variants generation process"""
    return