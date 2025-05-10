# /tmp/MetaGPT_fork_sela/metagpt/ext/sela/insights/instruction_generator.py
import json
from pathlib import Path
import traceback

# Import the globally configured DATA_CONFIG from utils.py
# No longer need to import load_data_config here as we'll use the global DATA_CONFIG
from metagpt.ext.sela.utils import DATA_CONFIG, clean_json_from_rsp 

DEFAULT_INSIGHTS_POOL_PATH = Path(__file__).parent / "fixed_insights.json"

class InstructionGenerator:
    def __init__(self, task_name: str, insights_pool_path: str = None): # Removed data_config_path argument
        self.task_name = task_name
        
        # Use the globally configured DATA_CONFIG from utils.py which has SELA path overrides
        self.runtime_data_config = DATA_CONFIG 
        
        if not self.runtime_data_config:
            raise ValueError("InstructionGenerator: DATA_CONFIG from utils.py is not available or empty.")

        datasets_metadata = self.runtime_data_config.get("datasets", {})
        if task_name not in datasets_metadata:
            # Try to construct a default metadata if not found, as dataset.py might have processed it
            # This is a fallback; ideally, datasets.yaml (loaded into DATA_CONFIG["datasets"]) should be complete.
            print(f"WARN: InstructionGenerator: Metadata for task '{task_name}' not found in DATA_CONFIG['datasets']. Using defaults.")
            self.dataset_metadata = {
                "target_col": "target", # Sensible default
                "metric": "f1"          # Sensible default
            }
        else:
            self.dataset_metadata = datasets_metadata[task_name]

        # dataset_info.json is created by dataset.py in the 'processed_datasets_output_dir'
        processed_output_dir = self.runtime_data_config.get("processed_datasets_output_dir")
        if not processed_output_dir:
            raise ValueError(f"InstructionGenerator: 'processed_datasets_output_dir' not found in DATA_CONFIG for task '{task_name}'")
        
        self.dataset_info_path = Path(processed_output_dir) / task_name / "dataset_info.json"
        print(f"INFO: InstructionGenerator: Attempting to load dataset_info.json from: {self.dataset_info_path}")
        
        try:
            with open(self.dataset_info_path, "r") as file:
                self.dataset_info = json.load(file)
            print(f"INFO: InstructionGenerator: Successfully loaded dataset_info.json for '{task_name}' from '{self.dataset_info_path}'")
        except FileNotFoundError:
            print(f"ERROR: InstructionGenerator: dataset_info.json NOT FOUND at '{self.dataset_info_path}'")
            # Provide more context from DATA_CONFIG for debugging
            print(f"DEBUG: InstructionGenerator: DATA_CONFIG['processed_datasets_output_dir'] = {processed_output_dir}")
            print(f"DEBUG: InstructionGenerator: task_name = {task_name}")
            raise
        except Exception as e:
            print(f"ERROR: InstructionGenerator: Failed to load or parse dataset_info.json from '{self.dataset_info_path}': {e}")
            traceback.print_exc()
            raise

        self.insights_pool_path = insights_pool_path or DEFAULT_INSIGHTS_POOL_PATH
        try:
            with open(self.insights_pool_path, "r") as file:
                self.insights_pool = json.load(file)
            print(f"INFO: InstructionGenerator: Successfully loaded insights pool from '{self.insights_pool_path}'")
        except FileNotFoundError:
            print(f"WARN: InstructionGenerator: Insights pool file not found at '{self.insights_pool_path}'. Using empty pool.")
            self.insights_pool = {} # Use an empty pool if the file is not found
        except Exception as e:
            print(f"ERROR: InstructionGenerator: Failed to load or parse insights_pool from '{self.insights_pool_path}': {e}")
            self.insights_pool = {}


    def get_dataset_insight(self, insight_name: str, top_k: int = 3):
        insights_str = ""
        dataset_insights = self.insights_pool.get(self.task_name, {})
        if insight_name in dataset_insights:
            insights = dataset_insights[insight_name]
            insights_str = f"\n## {insight_name.replace('_', ' ').title()}\n"
            for i, insight in enumerate(insights[:top_k]):
                insights_str += f"{i+1}. {insight}\n"
        return insights_str

    def get_solution_insight(self, solution: list, top_k: int = 3):
        insights_str = ""
        if "solution" in self.insights_pool:
            insights_str = "\n## Solution Insights\n"
            for i, insight in enumerate(self.insights_pool["solution"][:top_k]):
                insights_str += f"{i+1}. {insight}\n"
        return insights_str

    def get_instruction(self, solution: list):
        instruction = "Please complete the following task.\n"
        # instruction += self.get_dataset_insight("feature_engineering_insights")
        # instruction += self.get_dataset_insight("model_training_insights")
        # instruction += self.get_dataset_insight("data_preprocessing_insights")
        # instruction += self.get_solution_insight(solution) # This might not be used if solution insights are general
        return instruction

    def get_solution_instruction(self, solution: dict, prompt_type: str = "generate_code_from_nodes"):
        # This method seems specific to a different kind of "solution" object (dict)
        # and might not be directly used by the MCTS flow in the same way.
        # For now, ensure it runs without error if called.
        print(f"INFO: InstructionGenerator: get_solution_instruction called with prompt_type '{prompt_type}'")
        instruction = solution.get("instruction", "Please implement the described solution.") # Default instruction
        return instruction

# Example usage (for testing this file standalone, if needed)
if __name__ == '__main__':
    # This block would require DATA_CONFIG to be set up, typically by running another script first,
    # or by having data.yaml and datasets.yaml in the CWD or next to utils.py
    # and ensuring utils.py has run its global config logic.
    print("INFO: Running instruction_generator.py standalone test (requires DATA_CONFIG and processed dataset_info.json)")
    
    # For this test to run, DATA_CONFIG needs to be populated, especially 'processed_datasets_output_dir'
    # and 'datasets' (metadata).
    # Also, a sample dataset_info.json needs to exist in the expected output path.
    
    # Example:
    # Ensure DATA_CONFIG is populated as it would be by utils.py
    if not DATA_CONFIG or "processed_datasets_output_dir" not in DATA_CONFIG:
        print("WARN: Standalone test: DATA_CONFIG not fully populated by utils.py. Setting up mock DATA_CONFIG for test.")
        DATA_CONFIG["processed_datasets_output_dir"] = Path("./tmp_sela_output_test/sela_datasets_output")
        DATA_CONFIG["datasets_dir"] = Path("./tmp_sela_input_test") # Mock input
        DATA_CONFIG["datasets"] = {
            "sample_task": {"target_col": "label", "metric": "accuracy"}
        }
        sample_task_output_dir = DATA_CONFIG["processed_datasets_output_dir"] / "sample_task"
        sample_task_output_dir.mkdir(parents=True, exist_ok=True)
        mock_info_path = sample_task_output_dir / "dataset_info.json"
        if not mock_info_path.exists():
            with open(mock_info_path, "w") as f:
                json.dump({"name": "sample_task", "metadata": {"NumberOfClasses": 2}}, f)
            print(f"INFO: Created mock dataset_info.json at {mock_info_path}")

    try:
        if "sample_task" in DATA_CONFIG.get("datasets",{}):
            generator = InstructionGenerator(task_name="sample_task")
            print("\n--- Generated Instruction ---")
            print(generator.get_instruction(solution=[])) # Example solution
            print("\n--- Dataset Info Loaded ---")
            print(json.dumps(generator.dataset_info, indent=2))
        else:
            print("ERROR: Standalone test: 'sample_task' not in DATA_CONFIG['datasets']. Cannot run test.")

    except Exception as e:
        print(f"ERROR in InstructionGenerator standalone test: {e}")
        traceback.print_exc()