from ladder.engines import VerificationEngine, LLMEngine
from typing_extensions import Union
from loguru import logger
try:
    from scipy.optimize import minimize_scalar
    import numpy as np
    import sympy as sp
except ImportError:
    raise ImportError("NearestPointVerificationEngine requires scipy, numpy, and sympy to be installed. run `pip install scipy numpy sympy`")
import re

class NearestPointVerificationEngine(VerificationEngine):
    """ Custom Verification engine to select the nearest point on a curve / line to a reference point """
    
    def verify(self, problem_question: str, given_answer: str | tuple):
        # get correct answer 
        closest_point = self.get_correct_answer(problem_question)
        if not closest_point:
            return 0
        # Calculate difference between answer and closest point
        answer_tuple = self._ensure_tuple(given_answer)
        if not answer_tuple:
            return 0
        
        difference = self._calculate_difference(answer_tuple, closest_point)
        logger.success(f"difference: {difference}")
        
        return max(0, 1 - difference)
    
    def get_correct_answer(self, problem_question: str ):
        """
        Get the correct answer for the problem question
        """
        equation_str, reference_point = self._parse_problem_question(problem_question)
        if not equation_str or not reference_point:
            return None
        
        return self._closest_point_on_curve_tool(equation_str, reference_point)
    
    def _parse_problem_question(self, question_text: str) -> tuple[str, tuple]:
        """
        Extract equation and reference point from problem question text.
        """
        equation_pattern = r'y\s*=\s*([-\d\s\*\+\/\.\^xX()]+?)(?=\s+[a-zA-Z]|\s*\(.*,.*\)|$)'
        equation_match = re.search(equation_pattern, question_text)

        if not equation_match:
            return None, None

        equation_str = equation_match.group(1).strip()
        equation_str = self._normalize_equation(equation_str)

        point_pattern = r'\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)'
        point_match = re.search(point_pattern, question_text)

        if not point_match:
            if "origin" in question_text.lower():
                return equation_str, (0, 0)
            return None, None

        x_coord = float(point_match.group(1))
        y_coord = float(point_match.group(2))
        reference_point = (x_coord, y_coord)

        return equation_str, reference_point
    def _normalize_equation(self, equation_str: str) -> str:
        """
        Normalize equation string to be compatible with SymPy.
        Handles implicit multiplication like '2x' -> '2*x', '-0.5x' -> '-0.5*x'
        """
        # Remove extra whitespace
        equation_str = equation_str.strip()
        
        # Handle implicit multiplication patterns
        # Pattern: number followed by variable (like 2x, -0.5x, 3.14x)
        equation_str = re.sub(r'([+-]?\d*\.?\d+)([a-zA-Z])', r'\1*\2', equation_str)
        
        # Handle cases where there might be spaces around operators
        equation_str = re.sub(r'\s+', '', equation_str)  # Remove all spaces first
        
        # Add spaces around operators for readability (optional)
        equation_str = re.sub(r'([+-])', r' \1 ', equation_str)
        equation_str = re.sub(r'\s+', ' ', equation_str).strip()  # Clean up multiple spaces
        
        return equation_str
    
    def _ensure_tuple(self, answer: str | list| tuple) -> tuple:
        """
        Ensure the answer is a tuple format.
        Handle various input formats: tuple, list, string representation, etc.
        """

        if not answer:
            return None
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
            return None
 
    def _calculate_difference(self, answer_tuple: tuple, closest_point: tuple):
        """
        Calculate the Euclidean distance between answer tuple and closest point tuple.
        """
        x1, y1 = answer_tuple
        x2, y2 = closest_point
        
        # Euclidean distance
        difference = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return difference
    
    def _closest_point_on_curve_tool(self, equation_str: str, point: tuple) -> tuple:
        """
        Given a curve equation y = f(x) as string and a point (x0, y0),
        returns the closest point (x, y) on the curve to the point.

        Args:
            equation_str (str): Curve equation in terms of x, e.g. "-0.5*x**3 + 4*x**2 + 2*x + 5"
            point (tuple): Coordinates (x0, y0) of the reference point

        Returns:
            tuple: Closest point (x, y) on the curve
        """
        try:
            x = sp.symbols('x')
            expr_y = sp.sympify(equation_str)
            x0, y0 = point

            # Convert sympy expression to numerical function
            f_y = sp.lambdify(x, expr_y, 'numpy')

            # Define distance squared function
            def dist_sq(x_val):
                y_val = f_y(x_val)
                return (x_val - x0)**2 + (y_val - y0)**2

            # Minimize distance squared function over real numbers
            result = minimize_scalar(dist_sq)

            x_closest = result.x
            y_closest = f_y(x_closest)


            return (x_closest, y_closest)
            
        except Exception as e:
            logger.warning(f"Error in closest_point_on_curve_tool: {e}")
            return None
        


    
class PointResponseLLMEngine(LLMEngine):
    
    def generate(self, prompt, exported_format="str", *args, **kwargs):
        prompt = f"""
        You are Helpful assistant. Keep your answer short and to the point. dont show steps unless asked.
        <USER INPUT>
        {prompt}
        
        <YOUR RESPONSE>
        """
        response = self.lm(prompt, *args, **kwargs)
        if isinstance(response, list):
            response: str = response[0]
        
        # Extract point based on exported_format
        return self._extract_point(response)
    
    def _extract_point(self, text: str) -> Union[tuple[float, float], str]:
        """Extract the first coordinate point from text"""
        pattern = r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)'
        
        match = re.search(pattern, text)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            return (x, y)
        
        xy_pattern = r'x\s*=\s*(-?\d+(?:\.\d+)?),?\s*y\s*=\s*(-?\d+(?:\.\d+)?)'
        xy_match = re.search(xy_pattern, text, re.IGNORECASE)
        if xy_match:
            x, y = float(xy_match.group(1)), float(xy_match.group(2))
            return (x, y)
        
        return text 