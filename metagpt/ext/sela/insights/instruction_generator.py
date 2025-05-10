# /tmp/MetaGPT_fork_sela/metagpt/ext/sela/insights/instruction_generator.py
import json
from pathlib import Path
import traceback

# Import the globally configured DATA_CONFIG from utils.py
from metagpt.ext.sela.utils import DATA_CONFIG, clean_json_from_rsp # clean_json_from_rsp might be used by methods

# Path to a default/fixed insights pool if no specific one is provided
DEFAULT_INSIGHTS_POOL_PATH = Path(__file__).parent / "fixed_insights.json"

class InstructionGenerator:
    def __init__(self, state: dict, use_fixed_insights: bool, from_scratch: bool, insights_pool_path: str = None):
        if not isinstance(state, dict):
            raise TypeError(f"InstructionGenerator: 'state' argument must be a dictionary, got {type(state)}")
        
        self.task_name = state.get("task")
        if not self.task_name:
            raise ValueError("InstructionGenerator: 'task' key missing in state dictionary.")
            
        self.state_context = state  # Store the state if needed for other methods
        self.use_fixed_insights = use_fixed_insights
        self.from_scratch = from_scratch # Currently not used in generate_new_instructions, but stored

        # Use the global, SELA-configured DATA_CONFIG from utils.py
        # This DATA_CONFIG is expected to have 'processed_datasets_output_dir' and 'datasets' (metadata)
        self.runtime_data_config = DATA_CONFIG 
        
        if not self.runtime_data_config:
            raise ValueError("InstructionGenerator: Global DATA_CONFIG from utils.py is not available or empty.")

        datasets_metadata = self.runtime_data_config.get("datasets", {})
        if self.task_name not in datasets_metadata:
            print(f"WARN: InstructionGenerator: Metadata for task '{self.task_name}' not in DATA_CONFIG['datasets']. Using defaults.")
            self.dataset_metadata = {"target_col": "target", "metric": "f1_binary"} # Provide sensible defaults
        else:
            self.dataset_metadata = datasets_metadata[self.task_name]

        processed_output_dir = self.runtime_data_config.get("processed_datasets_output_dir")
        if not processed_output_dir:
            raise ValueError(f"InstructionGenerator: 'processed_datasets_output_dir' not found in DATA_CONFIG for task '{self.task_name}'")
        
        self.dataset_info_path = Path(processed_output_dir) / self.task_name / "dataset_info.json"
        print(f"INFO: InstructionGenerator: Initializing for task '{self.task_name}'. Attempting to load dataset_info.json from: {self.dataset_info_path}")
        
        try:
            with open(self.dataset_info_path, "r") as file:
                self.dataset_info = json.load(file)
            print(f"INFO: InstructionGenerator: Successfully loaded dataset_info.json for '{self.task_name}'")
        except FileNotFoundError:
            print(f"ERROR: InstructionGenerator: dataset_info.json NOT FOUND at '{self.dataset_info_path}'")
            raise
        except Exception as e:
            print(f"ERROR: InstructionGenerator: Failed to load or parse dataset_info.json from '{self.dataset_info_path}': {e}")
            traceback.print_exc()
            raise

        # Load insights pool
        self.insights_pool_path = insights_pool_path or DEFAULT_INSIGHTS_POOL_PATH # Use default if None
        try:
            with open(self.insights_pool_path, "r") as file:
                self.insights_pool = json.load(file)
            print(f"INFO: InstructionGenerator: Successfully loaded insights pool from '{self.insights_pool_path}'")
        except FileNotFoundError:
            print(f"WARN: InstructionGenerator: Insights pool file not found at '{self.insights_pool_path}'. Using empty insights pool.")
            self.insights_pool = {} 
        except Exception as e:
            print(f"ERROR: InstructionGenerator: Failed to load or parse insights_pool from '{self.insights_pool_path}': {e}")
            self.insights_pool = {}


    async def generate_new_instructions(self, task_id: int, original_instruction: str, max_num: int):
        """
        Generates a list of new instructions (actions) for child nodes.
        This is a basic implementation. A more advanced version might use an LLM
        or more sophisticated logic based on task_id, original_instruction, insights, etc.
        """
        print(f"INFO: InstructionGenerator: generate_new_instructions called for task_id={task_id}, max_num={max_num}")
        print(f"INFO: InstructionGenerator: Original instruction: '{original_instruction}'")
        
        new_instructions = []

        if self.use_fixed_insights and self.insights_pool:
            dataset_specific_insights = self.insights_pool.get(self.task_name, {})
            all_fixed_insights_for_task = []
            # Flatten insights from different categories for this task
            for category_insights in dataset_specific_insights.values():
                if isinstance(category_insights, list):
                    all_fixed_insights_for_task.extend(category_insights)
            
            if all_fixed_insights_for_task:
                # Create new instructions by combining original with fixed insights
                for i in range(min(max_num, len(all_fixed_insights_for_task))):
                    # Example: append insight to original. Could be more creative.
                    new_instructions.append(f"{original_instruction} One specific suggestion is to: {all_fixed_insights_for_task[i]}")
                print(f"INFO: InstructionGenerator: Generated {len(new_instructions)} instructions using fixed insights from pool.")

        # If not enough instructions generated from fixed insights, or if not using them,
        # add the original instruction and potentially simple variations.
        if not new_instructions: # Or if len(new_instructions) < max_num and want more variations
            new_instructions.append(original_instruction) # Always include the original or a base
            if max_num > 1 and len(new_instructions) < max_num :
                 new_instructions.append(f"{original_instruction} (Consider an alternative approach focusing on simplicity).")
            if max_num > 2 and len(new_instructions) < max_num :
                 new_instructions.append(f"{original_instruction} (Try to optimize for {self.dataset_metadata.get('metric', 'the target metric')}).")
            
            # Ensure we don't exceed max_num
            new_instructions = new_instructions[:max_num]
            print(f"INFO: InstructionGenerator: Generated/fallback to {len(new_instructions)} instructions.")
            
        if not new_instructions: # Should not happen if original_instruction is always added as fallback
             print("WARN: InstructionGenerator: No instructions were generated! Returning original instruction as a single option.")
             return [original_instruction]
             
        return new_instructions

    # Keep other methods from the original/previous version if they are used elsewhere or for reference
    def get_dataset_insight(self, insight_name: str, top_k: int = 3):
        insights_str = ""
        dataset_insights_from_pool = self.insights_pool.get(self.task_name, {})
        if insight_name in dataset_insights_from_pool:
            insights_list = dataset_insights_from_pool[insight_name]
            if insights_list: # Check if the list is not empty
                insights_str = f"\n## {insight_name.replace('_', ' ').title()}\n"
                for i, insight_text in enumerate(insights_list[:top_k]):
                    insights_str += f"{i+1}. {insight_text}\n"
        return insights_str

    def get_solution_insight(self, solution: list, top_k: int = 3): # 'solution' here might be list of actions
        # This method's utility depends on how 'solution' insights are structured in fixed_insights.json
        insights_str = ""
        if "solution_ideas" in self.insights_pool: # Example key
            insights_str = "\n## General Solution Ideas\n"
            for i, insight_text in enumerate(self.insights_pool["solution_ideas"][:top_k]):
                insights_str += f"{i+1}. {insight_text}\n"
        return insights_str

    def get_instruction(self, solution: list): # 'solution' might be the current path/actions taken
        # This method might be a more general instruction provider, not for expansion variations.
        # The 'generate_new_instructions' is specifically for MCTS expansion.
        instruction = "Please complete the current data interpretation task based on previous steps.\n"
        # Example: Add some context based on current state if needed
        # instruction += f"Current task focus: {self.task_name}.\n"
        # instruction += self.get_dataset_insight("feature_engineering_insights") # Example
        return instruction

    def get_solution_instruction(self, solution: dict, prompt_type: str = "generate_code_from_nodes"):
        # This seems to be for a specific use case where 'solution' is a dict with an 'instruction' key.
        print(f"INFO: InstructionGenerator: get_solution_instruction called (prompt_type: '{prompt_type}')")
        return solution.get("instruction", "Implement the solution as planned.")