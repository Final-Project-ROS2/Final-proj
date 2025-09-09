import google.generativeai as genai
import os
from dotenv import load_dotenv
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import json
import re

class GeminiVisionToPDDL:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        """
        Initialize the Gemini Vision to PDDL converter.
        
        Args:
            api_key (str): Google Gemini API key
            model_name (str): Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def describe_image(self, image_path, custom_prompt=None):
        """
        Use Gemini Vision to describe an image in detail.
        
        Args:
            image_path (str): Path to the image file
            custom_prompt (str): Custom prompt for image description
            
        Returns:
            str: Detailed image description
        """
        if custom_prompt is None:
            custom_prompt = """Describe this image with the most detail possible. Include:
            - All objects present and their positions
            - Spatial relationships between objects
            - Colors, shapes, and sizes
            - Any text or labels visible
            - The overall scene and context
            Focus on objects that could be manipulated or moved."""
        
        try:
            image = Image.open(image_path)
            response = self.model.generate_content([custom_prompt, image])
            return response.text
        except Exception as e:
            raise Exception(f"Error describing image: {str(e)}")
    
    def generate_pddl_from_description(self, image_description, instructions):
        """
        Generate PDDL domain and problem files from image description and instructions.
        
        Args:
            image_description (str): Detailed image description from Gemini
            instructions (str): Task instructions
            
        Returns:
            tuple: (domain_pddl, problem_pddl)
        """
        pddl_prompt = f"""
        Based on the following image description and task instructions, generate PDDL domain and problem files.

        IMAGE DESCRIPTION:
        {image_description}

        TASK INSTRUCTIONS:
        {instructions}

        Please generate:
        1. A PDDL domain file that defines the types, predicates, and actions needed
        2. A PDDL problem file that defines the initial state and goal

        Format your response as follows:
        DOMAIN:
        ```pddl
        [domain file content here]
        ```

        PROBLEM:
        ```pddl
        [problem file content here]
        ```

        Guidelines:
        - Use standard PDDL syntax
        - Include relevant object types (e.g., object, location, robot, etc.)
        - Define predicates for object properties and relationships
        - Include actions like pick, place, move, etc.
        - Set up initial state based on image description
        - Define goal state based on instructions
        - Use clear, descriptive names for objects and predicates
        """
        
        try:
            response = self.model.generate_content(pddl_prompt)
            return self._parse_pddl_response(response.text)
        except Exception as e:
            raise Exception(f"Error generating PDDL: {str(e)}")
    
    def _parse_pddl_response(self, response_text):
        """Parse the PDDL response to extract domain and problem files."""
        domain_match = re.search(r'DOMAIN:\s*```pddl\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        problem_match = re.search(r'PROBLEM:\s*```pddl\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        
        if not domain_match or not problem_match:
            raise Exception("Could not parse PDDL files from response")
        
        domain_pddl = domain_match.group(1).strip()
        problem_pddl = problem_match.group(1).strip()
        
        return domain_pddl, problem_pddl
    
    def save_pddl_files(self, domain_pddl, problem_pddl, output_dir="."):
        """
        Save PDDL files to disk.
        
        Args:
            domain_pddl (str): Domain PDDL content
            problem_pddl (str): Problem PDDL content
            output_dir (str): Directory to save files
            
        Returns:
            tuple: (domain_file_path, problem_file_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        domain_file = output_path / "domain.pddl"
        problem_file = output_path / "problem.pddl"
        
        with open(domain_file, 'w') as f:
            f.write(domain_pddl)
        
        with open(problem_file, 'w') as f:
            f.write(problem_pddl)
        
        return str(domain_file), str(problem_file)
    
    def run_fast_downward(self, domain_file, problem_file, output_dir=".", search_algorithm="astar(lmcut())"):
        """
        Execute Fast Downward planner on the PDDL files.
        
        Args:
            domain_file (str): Path to domain PDDL file
            problem_file (str): Path to problem PDDL file
            output_dir (str): Directory for output files
            search_algorithm (str): Search algorithm for Fast Downward
            
        Returns:
            dict: Planning results including plan, statistics, and status
        """
        output_path = Path(output_dir)
        plan_file = output_path / "plan.txt"
        log_file = output_path / "planner.log"
        
        # Fast Downward command
        # Note: Adjust the path to fast-downward.py based on your installation
        cmd = [
            "uv", "run", "./fast-downward/fast-downward.py",  # Adjust path as needed
            domain_file,
            problem_file,
            "--search", search_algorithm,
        ]
        
        try:
            # Run Fast Downward
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            planning_result = {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "plan": None,
                "plan_length": 0,
                "search_time": None
            }
            
            # Read plan if it exists
            if plan_file.exists():
                with open(plan_file, 'r') as f:
                    plan_content = f.read().strip()
                    planning_result["plan"] = plan_content
                    # Count plan steps (excluding comments and empty lines)
                    plan_steps = [line for line in plan_content.split('\n') 
                                if line.strip() and not line.startswith(';')]
                    planning_result["plan_length"] = len(plan_steps)
            
            # Extract search time from output
            time_match = re.search(r'Search time: ([\d.]+)s', result.stdout)
            if time_match:
                planning_result["search_time"] = float(time_match.group(1))
            
            return planning_result
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "return_code": -1,
                "stdout": "",
                "stderr": "Planning timed out after 5 minutes",
                "plan": None,
                "plan_length": 0,
                "search_time": None
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "return_code": -1,
                "stdout": "",
                "stderr": "Fast Downward not found. Please ensure it's installed and in PATH.",
                "plan": None,
                "plan_length": 0,
                "search_time": None
            }
        except Exception as e:
            return {
                "status": "error",
                "return_code": -1,
                "stdout": "",
                "stderr": f"Error running planner: {str(e)}",
                "plan": None,
                "plan_length": 0,
                "search_time": None
            }
    
    def process_image_to_plan(self, image_path, instructions, output_dir="output", 
                            custom_prompt=None, search_algorithm="astar(lmcut())"):
        """
        Complete pipeline: image description -> PDDL generation -> planning.
        
        Args:
            image_path (str): Path to input image
            instructions (str): Task instructions
            output_dir (str): Output directory for all files
            custom_prompt (str): Custom prompt for image description
            search_algorithm (str): Search algorithm for Fast Downward
            
        Returns:
            dict: Complete results including description, PDDL, and plan
        """
        print("Step 1: Analyzing image...")
        description = self.describe_image(image_path, custom_prompt)
        
        print("Step 2: Generating PDDL files...")
        domain_pddl, problem_pddl = self.generate_pddl_from_description(description, instructions)
        
        print("Step 3: Saving PDDL files...")
        domain_file, problem_file = self.save_pddl_files(domain_pddl, problem_pddl, output_dir)
        
        print("Step 4: Running planner...")
        planning_result = self.run_fast_downward(domain_file, problem_file, output_dir, search_algorithm)
        
        return {
            "image_description": description,
            "domain_pddl": domain_pddl,
            "problem_pddl": problem_pddl,
            "domain_file": domain_file,
            "problem_file": problem_file,
            "planning_result": planning_result
        }

def main():
    """Example usage of the GeminiVisionToPDDL class."""
    
    load_dotenv()
    # Initialize with your Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")  # Set this environment variable
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    converter = GeminiVisionToPDDL(api_key)
    
    # Example usage
    image_path = "example_image.jpg"  # Replace with your image path
    instructions = """
    I want a stack consisting of block A on top of B on top of D.
    Block C is place on the side.
    """
    
    try:
        results = converter.process_image_to_plan(
            image_path=image_path,
            instructions=instructions,
            output_dir="planning_output"
        )
        
        print("\n=== RESULTS ===")
        print(f"Image Description Length: {len(results['image_description'])} characters")
        print(f"")
        print(f"Planning Status: {results['planning_result']['status']}")
        
        if results['planning_result']['plan']:
            print(f"Plan Length: {results['planning_result']['plan_length']} steps")
            print("\nGenerated Plan:")
            print(results['planning_result']['plan'])
        else:
            print("No plan generated")
            print(f"Error: {results['planning_result']['stderr']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()