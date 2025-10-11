from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import subprocess
from pathlib import Path
from PIL import Image
import re
import base64
from typing import List, Dict, Any, Optional, Tuple
import io
from instructions import instructions

class PDDLGenerationResult(BaseModel):
    """Result model for PDDL generation."""
    domain_pddl: str = Field(description="Generated PDDL domain file content")
    problem_pddl: str = Field(description="Generated PDDL problem file content")
    reasoning: str = Field(description="Reasoning behind the PDDL generation")

class PlanningResult(BaseModel):
    """Result model for planning execution."""
    status: str = Field(description="Planning status: success, failed, timeout, or error")
    return_code: int = Field(description="Return code from planner")
    stdout: str = Field(description="Standard output from planner")
    stderr: str = Field(description="Standard error from planner")
    plan: Optional[str] = Field(description="Generated plan if successful")
    plan_length: int = Field(description="Number of steps in the plan")
    search_time: Optional[float] = Field(description="Search time in seconds")

class LangChainGeminiVisionToPDDL:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0):
        """
        Initialize the LangChain-based Gemini Vision to PDDL converter.
        
        Args:
            api_key (str): Google Gemini API key
            model_name (str): Gemini model to use
            temperature (float): Temperature for LLM responses
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize vision model for image description
        self.vision_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
        
        # Initialize text model for PDDL generation (with tool calling capabilities)
        self.pddl_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
        
        # Initialize tools for PDDL generation
        self.tools = self._initialize_tools()
        
        # Create agent for PDDL generation with tool calling
        self.pddl_agent = self._create_pddl_agent()
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize tools that the PDDL generation LLM can use."""
        
        @tool
        def clarify_object_properties(object_name: str, property_type: str) -> str:
            """
            Request clarification about object properties from the user.
            
            Args:
                object_name: Name of the object to clarify
                property_type: Type of property (e.g., 'color', 'size', 'material', 'position')
            
            Returns:
                Clarification about the object property
            """
            print(f"\n🤖 TOOL CALL: Requesting clarification about {object_name}'s {property_type}")
            response = input(f"Please clarify the {property_type} of {object_name}: ")
            return f"The {property_type} of {object_name} is: {response}"
        
        @tool
        def get_spatial_relationships(object1: str, object2: str) -> str:
            """
            Get detailed spatial relationship between two objects.
            
            Args:
                object1: First object
                object2: Second object
            
            Returns:
                Detailed spatial relationship description
            """
            print(f"\n🤖 TOOL CALL: Requesting spatial relationship between {object1} and {object2}")
            response = input(f"Describe the spatial relationship between {object1} and {object2}: ")
            return f"Spatial relationship: {response}"
        
        @tool
        def clarify_instructions(instruction_part: str) -> str:
            """
            Request clarification about ambiguous instructions.
            
            Args:
                instruction_part: The part of instructions that needs clarification
            
            Returns:
                Clarified instruction
            """
            print(f"\n🤖 TOOL CALL: Requesting clarification about instruction")
            print(f"Ambiguous instruction: {instruction_part}")
            response = input("Please clarify this instruction: ")
            return f"Clarified instruction: {response}"
        
        @tool
        def validate_pddl_syntax(pddl_content: str, file_type: str) -> str:
            """
            Validate PDDL syntax for domain or problem files.
            
            Args:
                pddl_content: PDDL file content to validate
                file_type: Either 'domain' or 'problem'
            
            Returns:
                Validation result and suggestions
            """
            # Basic PDDL syntax validation
            required_keywords = {
                'domain': ['define', 'domain', ':requirements', ':types', ':predicates', ':action'],
                'problem': ['define', 'problem', ':domain', ':objects', ':init', ':goal']
            }
            
            missing_keywords = []
            for keyword in required_keywords[file_type]:
                if keyword not in pddl_content.lower():
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                return f"VALIDATION FAILED: Missing required keywords: {', '.join(missing_keywords)}"
            else:
                return "VALIDATION PASSED: Basic PDDL syntax appears correct"
        
        @tool
        def get_domain_specific_actions(domain: str) -> str:
            """
            Get suggestions for domain-specific actions based on the task domain.
            
            Args:
                domain: The domain type (e.g., 'blocks_world', 'logistics', 'kitchen', 'tabletop')
            
            Returns:
                List of suggested actions for the domain
            """
            domain_actions = {
                'blocks_world': ['pick-up', 'put-down', 'stack', 'unstack'],
                'logistics': ['load', 'unload', 'drive', 'fly'],
                'kitchen': ['pick', 'place', 'open', 'close', 'pour', 'mix'],
                'tabletop': ['grasp', 'release', 'move', 'push', 'pull'],
                'general': ['pick', 'place', 'move', 'push', 'pull', 'grasp', 'release']
            }
            
            actions = domain_actions.get(domain.lower(), domain_actions['general'])
            return f"Suggested actions for {domain}: {', '.join(actions)}"
        
        return [
            clarify_object_properties,
            get_spatial_relationships,
            clarify_instructions,
            validate_pddl_syntax,
            # get_domain_specific_actions
        ]
    
    def _create_pddl_agent(self) -> AgentExecutor:
        """Create an agent with tool calling capabilities for PDDL generation."""
        
        system_message = """You are an expert in PDDL (Planning Domain Definition Language) generation. 
        Your task is to analyze image descriptions and task instructions to generate valid PDDL domain and problem files.

        When generating PDDL:
        1. Use standard PDDL syntax
        2. Include relevant object types (object, location, robot, etc.)
        3. Define predicates for object properties and relationships
        4. Include appropriate actions (pick, place, move, etc.)
        5. Set up initial state based on image description
        6. Define goal state based on instructions

        You have access to tools to:
        - Clarify ambiguous object properties
        - Get spatial relationships between objects
        - Clarify ambiguous instructions
        - Validate PDDL syntax

        Use these tools when you need more information or want to validate your output.
        Always provide both domain and problem files in your final response.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.pddl_llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=10)
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Gemini Vision."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def describe_image(self, image_path: str, custom_prompt: Optional[str] = None) -> str:
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
            - All objects present and their positions (be specific about locations)
            - Spatial relationships between objects (on, next to, behind, etc.)
            - Colors, shapes, and sizes of objects
            - Any text or labels visible
            - The overall scene and context (table, floor, workspace, etc.)
            - Object properties that would be relevant for manipulation (graspable, stackable, etc.)
            
            Focus on objects that could be manipulated or moved in a robotic task.
            Use precise spatial language and mention any containers, surfaces, or boundaries."""
        
        try:
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": custom_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"
                        }
                    }
                ]
            )
            
            response = self.vision_llm.invoke([message])
            return response.content
        except Exception as e:
            raise Exception(f"Error describing image: {str(e)}")
    
    def generate_pddl_from_description(self, image_description: str, instructions: str, 
                                     interactive: bool = False) -> PDDLGenerationResult:
        """
        Generate PDDL domain and problem files using the agent with tool calling.
        
        Args:
            image_description (str): Detailed image description from Gemini
            instructions (str): Task instructions
            interactive (bool): Whether to enable interactive tool calling
            
        Returns:
            PDDLGenerationResult: Generated PDDL files and reasoning
        """
        
        pddl_input = f"""
        Based on the following image description and task instructions, generate PDDL domain and problem files.

        IMAGE DESCRIPTION:
        {image_description}

        TASK INSTRUCTIONS:
        {instructions}

        Please analyze the scene and generate:
        1. A PDDL domain file that defines the types, predicates, and actions needed
        2. A PDDL problem file that defines the initial state and goal

        If you need clarification about objects, their properties, spatial relationships, or instructions, 
        use the available tools to get more information.

        Format your final response as follows:
        REASONING:
        [Explain your reasoning for the PDDL design]

        DOMAIN:
        ```pddl
        [domain file content here]
        ```

        PROBLEM:
        ```pddl
        [problem file content here]
        ```
        """
        
        try:
            if interactive:
                print("\nStarting PDDL generation with interactive tools...")
                print("The AI may ask for clarifications during generation.\n")
            
            response = self.pddl_agent.invoke({"input": pddl_input})
            response_text = response["output"]
            
            return self._parse_pddl_response(response_text)
        except Exception as e:
            raise Exception(f"Error generating PDDL: {str(e)}")
    
    def _parse_pddl_response(self, response_text: str) -> PDDLGenerationResult:
        """Parse the PDDL response to extract domain and problem files."""
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=DOMAIN:|$)', response_text, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Extract PDDL files
        domain_match = re.search(r'DOMAIN:\s*```pddl\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        problem_match = re.search(r'PROBLEM:\s*```pddl\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        
        if not domain_match or not problem_match:
            raise Exception("Could not parse PDDL files from response")
        
        domain_pddl = domain_match.group(1).strip()
        problem_pddl = problem_match.group(1).strip()
        
        return PDDLGenerationResult(
            domain_pddl=domain_pddl,
            problem_pddl=problem_pddl,
            reasoning=reasoning
        )
    
    def save_pddl_files(self, pddl_result: PDDLGenerationResult, output_dir: str = ".") -> Tuple[str, str]:
        """
        Save PDDL files to disk.
        
        Args:
            pddl_result (PDDLGenerationResult): PDDL generation result
            output_dir (str): Directory to save files
            
        Returns:
            tuple: (domain_file_path, problem_file_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        domain_file = output_path / "domain.pddl"
        problem_file = output_path / "problem.pddl"
        reasoning_file = output_path / "reasoning.txt"
        
        with open(domain_file, 'w') as f:
            f.write(pddl_result.domain_pddl)
        
        with open(problem_file, 'w') as f:
            f.write(pddl_result.problem_pddl)
        
        with open(reasoning_file, 'w') as f:
            f.write(pddl_result.reasoning)
        
        return str(domain_file), str(problem_file)
    
    def run_fast_downward(self, domain_file: str, problem_file: str, output_dir: str = ".", 
                         search_algorithm: str = "astar(lmcut())") -> PlanningResult:
        """
        Execute Fast Downward planner on the PDDL files.
        
        Args:
            domain_file (str): Path to domain PDDL file
            problem_file (str): Path to problem PDDL file
            output_dir (str): Directory for output files
            search_algorithm (str): Search algorithm for Fast Downward
            
        Returns:
            PlanningResult: Planning results including plan, statistics, and status
        """
        output_path = Path(".")
        plan_file = output_path / "sas_plan"
        
        # Fast Downward command
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
            planning_result = PlanningResult(
                status="success" if result.returncode == 0 else "failed",
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                plan=None,
                plan_length=0,
                search_time=None
            )
            
            # Read plan if it exists
            if plan_file.exists():
                with open(plan_file, 'r') as f:
                    plan_content = f.read().strip()
                    planning_result.plan = plan_content
                    # Count plan steps (excluding comments and empty lines)
                    plan_steps = [line for line in plan_content.split('\n') 
                                if line.strip() and not line.startswith(';')]
                    planning_result.plan_length = len(plan_steps)
            
            # Extract search time from output
            time_match = re.search(r'Search time: ([\d.]+)s', result.stdout)
            if time_match:
                planning_result.search_time = float(time_match.group(1))
            
            return planning_result
            
        except subprocess.TimeoutExpired:
            return PlanningResult(
                status="timeout",
                return_code=-1,
                stdout="",
                stderr="Planning timed out after 5 minutes",
                plan=None,
                plan_length=0,
                search_time=None
            )
        except FileNotFoundError:
            return PlanningResult(
                status="error",
                return_code=-1,
                stdout="",
                stderr="Fast Downward not found. Please ensure it's installed and in PATH.",
                plan=None,
                plan_length=0,
                search_time=None
            )
        except Exception as e:
            return PlanningResult(
                status="error",
                return_code=-1,
                stdout="",
                stderr=f"Error running planner: {str(e)}",
                plan=None,
                plan_length=0,
                search_time=None
            )
    
    def process_image_to_plan(self, image_path: str, instructions: str, output_dir: str = "output", 
                            custom_prompt: Optional[str] = None, search_algorithm: str = "astar(lmcut())",
                            interactive: bool = False) -> Dict[str, Any]:
        """
        Complete pipeline: image description -> PDDL generation -> planning.
        
        Args:
            image_path (str): Path to input image
            instructions (str): Task instructions
            output_dir (str): Output directory for all files
            custom_prompt (str): Custom prompt for image description
            search_algorithm (str): Search algorithm for Fast Downward
            interactive (bool): Enable interactive tool calling for PDDL generation
            
        Returns:
            dict: Complete results including description, PDDL, and plan
        """
        print("Step 1: Analyzing image with Gemini Vision...")
        description = self.describe_image(image_path, custom_prompt)
        
        print("\nStep 2: Generating PDDL files with LangChain agent...")
        pddl_result = self.generate_pddl_from_description(description, instructions, interactive)
        
        print("\nStep 3: Saving PDDL files...")
        domain_file, problem_file = self.save_pddl_files(pddl_result, output_dir)
        
        print("\nStep 4: Running Fast Downward planner...")
        planning_result = self.run_fast_downward(domain_file, problem_file, output_dir, search_algorithm)
        
        return {
            "image_description": description,
            "pddl_result": pddl_result,
            "domain_file": domain_file,
            "problem_file": problem_file,
            "planning_result": planning_result
        }

def main():
    """Example usage of the LangChain-based Gemini Vision to PDDL converter."""
    
    load_dotenv()
    # Initialize with your Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    converter = LangChainGeminiVisionToPDDL(api_key)
    
    # Example usage
    image_path = "./images/example_image_1.jpg"
    
    try:
        # Run with interactive tool calling enabled
        results = converter.process_image_to_plan(
            image_path=image_path,
            instructions=instructions[1][1],
            output_dir="planning_output",
            interactive=True  # Enable interactive clarifications
        )
        
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        # print(f"Image Description: {results['image_description']}")
        # print(f"PDDL Reasoning: {results['pddl_result'].reasoning}")
        print(f"Planning Status: {results['planning_result'].status}")
        
        if results['planning_result'].plan:
            print(f"Plan Length: {results['planning_result'].plan_length} steps")
            print(f"Search Time: {results['planning_result'].search_time}s")
            print("\nGenerated Plan:")
            print(results['planning_result'].plan)
        else:
            print("No plan generated")
            print(f"Error: {results['planning_result'].stderr}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()