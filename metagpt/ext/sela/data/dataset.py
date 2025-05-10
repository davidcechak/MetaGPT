import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback

import openml
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Attempt to import SolutionDesigner and DATA_CONFIG.
# DATA_CONFIG will be the one from your project's utils.py (metagpt/ext/sela/utils.py),
# which primarily loads data.yaml and datasets.yaml.
# The __main__ block below will then override/ensure specific keys for this script's execution.
try:
    from metagpt.ext.sela.insights.solution_designer import SolutionDesigner
    from metagpt.ext.sela.utils import DATA_CONFIG
except ImportError as e:
    print(f"WARN: dataset.py: Initial import of SELA modules (SolutionDesigner or DATA_CONFIG from utils) failed: {e}. Using placeholders until __main__ configuration.")
    SolutionDesigner = object # Placeholder; __main__ will check this.
    DATA_CONFIG = {}        # Placeholder; __main__ will populate/override.

# --- SELA-specific constants defined directly in dataset.py ---
# This makes dataset.py self-reliant for these, as requested,
# since the provided utils.py does not define them.

BASE_USER_REQUIREMENT = """
This is a {datasetname} dataset. Your goal is to predict the target column `{target_col}`.
Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. 
Report {metric} on the eval data. Do not plot or make any visualizations.
"""

USE_AG = """
- Please use autogluon for model training with presets='medium_quality', time_limit=None, give dev dataset to tuning_data, and use right eval_metric.
"""

TEXT_MODALITY = """
- You could use models from transformers library for this text dataset.
- Use gpu if available for faster training.
"""

IMAGE_MODALITY = """
- You could use models from transformers/torchvision library for this image dataset.
- Use gpu if available for faster training.
"""

STACKING = """
- To avoid overfitting, train a weighted ensemble model such as StackingClassifier or StackingRegressor.
- You could do some quick model prototyping to see which models work best and then use them in the ensemble. 
"""

SPECIAL_INSTRUCTIONS = {"ag": USE_AG, "stacking": STACKING, "text": TEXT_MODALITY, "image": IMAGE_MODALITY}

DI_INSTRUCTION = """
## Attention
1. Please do not leak the target label in any form during training.
2. Test set does not have the target column.
3. When conducting data exploration or analysis, print out the results of your findings.
4. You should perform transformations on train, dev, and test sets at the same time (it's a good idea to define functions for this and avoid code repetition).
5. When scaling or transforming features, make sure the target column is not included.
6. You could utilize dev set to validate and improve model training. {special_instruction}

## Saving Dev and Test Predictions
1. Save the prediction results of BOTH the dev set and test set in `dev_predictions.csv` and `test_predictions.csv` respectively in the output directory. 
- Both files should contain a single column named `target` with the predicted values.
2. Make sure the prediction results are in the same format as the target column in the original training set. 
- For instance, if the original target column is a list of string, the prediction results should also be strings.

## Output Performance
Print the train and dev set performance in the last step.

# Output dir
{output_dir}
"""

TASK_PROMPT = """
# User requirement
{user_requirement}
{additional_instruction}
# Data dir
train set (with labels): {train_path}
dev set (with labels): {dev_path}
test set (without labels): {test_path}
dataset description: {data_info_path} (During EDA, you can use this file to get additional information about the dataset)
"""
# --- End of SELA-specific constants ---

SEED = 100
TRAIN_TEST_SPLIT = 0.8
TRAIN_DEV_SPLIT = 0.75

OPENML_DATASET_IDS = []
CUSTOM_DATASETS = []
DSAGENT_DATASETS = []


def get_split_dataset_path(dataset_name, config):
    processed_output_dir = config.get("processed_datasets_output_dir")
    if not processed_output_dir:
        raise ValueError("'processed_datasets_output_dir' not in config for get_split_dataset_path")
    data_path = Path(processed_output_dir) / dataset_name
    split_datasets = {
        "train": str(data_path / "split_train.csv"),
        "dev": str(data_path / "split_dev.csv"),
        "dev_wo_target": str(data_path / "split_dev_wo_target.csv"),
        "dev_target": str(data_path / "split_dev_target.csv"),
        "test": str(data_path / "split_test.csv"),
        "test_wo_target": str(data_path / "split_test_wo_target.csv"),
        "test_target": str(data_path / "split_test_target.csv"),
    }
    return split_datasets

def get_user_requirement(task_name, config):
    datasets_metadata = config.get("datasets", {}) # DATA_CONFIG["datasets"] from utils.py (datasets.yaml)
    if task_name in datasets_metadata:
        dataset_meta = datasets_metadata[task_name]
        user_req = dataset_meta.get("user_requirement")
        if not user_req:
             print(f"WARN: 'user_requirement' for '{task_name}' not in datasets.yaml. Generating default.")
             metric = dataset_meta.get("metric", "f1_binary")
             target_col = dataset_meta.get("target_col", "target")
             return BASE_USER_REQUIREMENT.format(datasetname=task_name, target_col=target_col, metric=metric)
        return user_req
    else:
        print(f"WARN: Metadata for dataset '{task_name}' not found in datasets.yaml. Generating generic user_requirement.")
        return BASE_USER_REQUIREMENT.format(datasetname=task_name, target_col="target_placeholder", metric="metric_placeholder")

def save_datasets_dict_to_yaml(datasets_dict, name_or_path="datasets.yaml"):
    # This saves metadata about processed datasets (name, requirements, etc.)
    # It's intended to update the datasets.yaml used by SELA.
    target_path_str = name_or_path
    try:
        from metagpt.ext.sela.utils import DEFAULT_DATASETS_YAML_PATH as sela_default_yaml
        # Ensure the path from utils is treated as absolute or relative to the correct base
        # For now, assuming sela_default_yaml is the desired full path.
        target_path_str = str(sela_default_yaml)
        print(f"INFO: Saving datasets metadata to standard SELA path: {target_path_str}")
    except ImportError:
        # Fallback if utils or the constant is not available
        work_dir_path = DATA_CONFIG.get("work_dir", ".")
        target_path_str = str(Path(work_dir_path) / Path(name_or_path).name) # Ensure it's just filename if relative
        print(f"WARN: DEFAULT_DATASETS_YAML_PATH from utils not found. Saving datasets metadata to: {target_path_str}")

    final_path = Path(target_path_str)
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w") as file:
            yaml.dump(datasets_dict, file, default_flow_style=False, sort_keys=False)
        print(f"INFO: Successfully saved datasets metadata to {final_path}")
    except Exception as e:
        print(f"ERROR: Could not save datasets metadata to {final_path}: {e}")
        traceback.print_exc()


def create_dataset_dict(dataset: 'ExpDataset'):
    return {
        "dataset": dataset.name,
        "user_requirement": dataset.create_base_requirement(),
        "metric": dataset.get_metric(),
        "target_col": dataset.target_col,
    }

def generate_di_instruction(agent_run_output_dir, special_instruction_code): # Changed param name
    prompt_text = ""
    if special_instruction_code and special_instruction_code in SPECIAL_INSTRUCTIONS: # Changed param name
        prompt_text = SPECIAL_INSTRUCTIONS[special_instruction_code] # Changed param name
    elif special_instruction_code:
        print(f"WARN: Special instruction code '{special_instruction_code}' not in SPECIAL_INSTRUCTIONS dict.")
    return DI_INSTRUCTION.format(output_dir=str(agent_run_output_dir), special_instruction=prompt_text)

def generate_task_requirement(task_name, data_config_runtime, is_di=True, special_instruction=None): # Changed param name
    req_text = get_user_requirement(task_name, data_config_runtime)
    split_file_paths = get_split_dataset_path(task_name, data_config_runtime) # Uses processed_datasets_output_dir

    # Paths for the prompt point to where dataset.py *saved* the splits
    train_p, dev_p, test_p_infer = split_file_paths["train"], split_file_paths["dev"], split_file_paths["test_wo_target"]
    
    # output_dir for the *agent's run specific outputs*, not where dataset.py saves splits
    agent_output_directory = Path(data_config_runtime.get("work_dir", ".")) / task_name 
    
    # data_info_path points to where dataset_info.json was *saved* by dataset.py
    info_file_p = Path(data_config_runtime.get("processed_datasets_output_dir", ".")) / task_name / "dataset_info.json"

    additional_instr_text = generate_di_instruction(str(agent_output_directory), special_instruction) if is_di else ""

    final_prompt = TASK_PROMPT.format(
        user_requirement=req_text, train_path=str(train_p), dev_path=str(dev_p),
        test_path=str(test_p_infer), additional_instruction=additional_instr_text, data_info_path=str(info_file_p)
    )
    print(f"INFO: Generated task requirement for '{task_name}'. Data paths point to processed files.")
    # print(f"Prompt details: \nTrain: {train_p}\nDev: {dev_p}\nTest: {test_p_infer}\nInfo: {info_file_p}")
    return final_prompt

class ExpDataset:
    description: str = None
    metadata: dict = None
    target_col: str = None
    name: str = None

    def __init__(self, name, input_datasets_base_dir, **kwargs):
        self.name = name
        self.input_datasets_base_dir = Path(input_datasets_base_dir)
        self.target_col = kwargs.get("target_col", "target")
        self.force_update = kwargs.get("force_update", False)

        self.input_specific_dataset_path = self.input_datasets_base_dir / self.name
        
        processed_output_base = Path(DATA_CONFIG.get("processed_datasets_output_dir", Path(DATA_CONFIG.get("work_dir", ".")) / "sela_datasets_output_fallback"))
        self.output_specific_dataset_path = processed_output_base / self.name
        self.output_specific_dataset_path.mkdir(parents=True, exist_ok=True)

        print(f"INFO: ExpDataset '{self.name}': Input from '{self.input_specific_dataset_path}', Output to '{self.output_specific_dataset_path}'")
        self.save_dataset() # Pass target_col via self

    def check_dataset_exists(self): # Checks for *output* split files
        fnames = ["split_train.csv", "split_dev.csv", "split_test.csv"]
        return all((self.output_specific_dataset_path / f).exists() for f in fnames)

    def check_datasetinfo_exists(self): # Checks for *output* info file
        return (self.output_specific_dataset_path / "dataset_info.json").exists()

    def get_raw_dataset(self): # Reads from *input* path
        path_to_check = self.input_specific_dataset_path
        direct_train = path_to_check / "train.csv"
        direct_test = path_to_check / "test.csv"
        raw_subdir = path_to_check / "raw"
        raw_subdir_train = raw_subdir / "train.csv"
        raw_subdir_test = raw_subdir / "test.csv"
        train_df, test_df = None, None

        if direct_train.exists():
            print(f"INFO: Found raw train.csv: {direct_train}")
            train_df = pd.read_csv(direct_train)
            if direct_test.exists(): print(f"INFO: Found raw test.csv: {direct_test}"); test_df = pd.read_csv(direct_test)
        elif raw_subdir_train.exists():
            print(f"INFO: Found raw train.csv in 'raw' subdir: {raw_subdir_train}")
            train_df = pd.read_csv(raw_subdir_train)
            if raw_subdir_test.exists(): print(f"INFO: Found raw test.csv in 'raw' subdir: {raw_subdir_test}"); test_df = pd.read_csv(raw_subdir_test)
        else:
            raise FileNotFoundError(f"Raw `train.csv` for '{self.name}' not found in {path_to_check} or {raw_subdir}")
        return train_df, test_df

    def get_dataset_info(self):
        raw_train_df, _ = self.get_raw_dataset()
        if raw_train_df is None: raise RuntimeError(f"Cannot get info for {self.name}: raw train data missing.")
        if not self.target_col or self.target_col not in raw_train_df.columns:
             print(f"WARN: Target '{self.target_col}' not in {self.name} columns: {raw_train_df.columns}. Using last or placeholder.")
             self.target_col = raw_train_df.columns[-1] if not raw_train_df.empty else "unknown_target_col"
        
        unique_classes = raw_train_df[self.target_col].nunique() if self.target_col in raw_train_df else 0
        return {
            "name": self.name, "description": self.description or f"Dataset: {self.name}",
            "target_col": self.target_col,
            "metadata": {"NumberOfClasses": unique_classes, "NumberOfFeatures": raw_train_df.shape[1],
                         "NumberOfInstances": raw_train_df.shape[0],
                         "NumberOfInstancesWithMissingValues": int(raw_train_df.isnull().any(axis=1).sum()),
                         "NumberOfMissingValues": int(raw_train_df.isnull().sum().sum()),
                         "NumberOfNumericFeatures": raw_train_df.select_dtypes(include=["number"]).shape[1],
                         "NumberOfSymbolicFeatures": raw_train_df.select_dtypes(include=["object"]).shape[1]},
            "df_head": raw_train_df.head().to_string(index=False)
        }

    def get_df_head(self, df): return df.head().to_string(index=False)

    def get_metric(self):
        info = self.get_dataset_info(); num_cls = info["metadata"].get("NumberOfClasses", 0)
        if num_cls == 2: return "f1_binary"
        elif 2 < num_cls <= 200: return "f1_weighted"
        return "rmse"

    def create_base_requirement(self):
        return BASE_USER_REQUIREMENT.format(datasetname=self.name, target_col=self.target_col, metric=self.get_metric())

    def save_dataset(self): # Removed target_col arg, uses self.target_col
        if not self.target_col: raise ValueError(f"Target column for {self.name} must be set before saving.")
        df_train_raw, df_test_raw_opt = self.get_raw_dataset()
        if df_train_raw is None: print(f"ERROR: Raw train data for {self.name} is None. Cannot save."); return

        if not self.check_dataset_exists() or self.force_update:
            print(f"INFO: Processing/saving dataset '{self.name}' to '{self.output_specific_dataset_path}'")
            self.split_and_save(df_train_raw, self.target_col, test_df_raw=df_test_raw_opt)
        else:
            print(f"INFO: Processed splits for '{self.name}' exist at '{self.output_specific_dataset_path}'. Skipping.")

        if not self.check_datasetinfo_exists() or self.force_update:
            print(f"INFO: Saving dataset_info.json for '{self.name}' to '{self.output_specific_dataset_path}'")
            try: self.save_datasetinfo(self.get_dataset_info())
            except Exception as e: print(f"ERROR: Saving dataset_info for {self.name} failed: {e}"); traceback.print_exc()
        else:
            print(f"INFO: dataset_info.json for '{self.name}' exists at '{self.output_specific_dataset_path}'. Skipping.")

    def save_datasetinfo(self, info_dict):
        path = self.output_specific_dataset_path / "dataset_info.json"
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(info_dict, f, indent=4, ensure_ascii=False)
            print(f"INFO: Saved dataset_info.json to {path}")
        except Exception as e: print(f"ERROR: Writing dataset_info.json to {path} failed: {e}")

    def save_split_datasets(self, df, label, target): # Saves to output_specific_dataset_path
        out_dir = self.output_specific_dataset_path; main_f = out_dir / f"split_{label}.csv"
        df.to_csv(main_f, index=False); print(f"INFO: Saved {main_f}")
        if target and target in df.columns:
            df.drop(columns=[target]).to_csv(out_dir / f"split_{label}_wo_target.csv", index=False)
            target_df = df[[target]].copy()
            if target != "target": target_df.rename(columns={target: "target"}, inplace=True)
            target_df.to_csv(out_dir / f"split_{label}_target.csv", index=False)
        elif target: print(f"WARN: Target '{target}' not in df for split '{label}'.")

    def split_and_save(self, train_raw, target, test_df_raw=None):
        if not target or target not in train_raw.columns:
            raise ValueError(f"Target '{target}' invalid for {self.name}")
        
        train_main, test_with_target = train_raw, test_df_raw
        stratify_opt = lambda df: df[target] if df[target].nunique() > 1 and len(df) >= 2 * df[target].nunique() else None

        if test_df_raw is None:
            print(f"INFO: Splitting {self.name} raw train for train/test sets.")
            train_main, test_with_target = train_test_split(train_raw, test_size=(1-TRAIN_TEST_SPLIT), random_state=SEED, stratify=stratify_opt(train_raw))
        else:
            print(f"INFO: Using provided raw test data for {self.name}.")
            if target not in test_with_target.columns:
                print(f"WARN: Target '{target}' not in provided raw test data for {self.name}. Adding as NA."); test_with_target[target] = pd.NA
        
        final_train, final_dev = train_test_split(train_main, test_size=(1-TRAIN_DEV_SPLIT), random_state=SEED, stratify=stratify_opt(train_main))
        self.save_split_datasets(final_train, "train", target)
        self.save_split_datasets(final_dev, "dev", target)
        self.save_split_datasets(test_with_target, "test", target)

class OpenMLExpDataset(ExpDataset):
    def __init__(self, _, input_base_dir, dataset_id, **kwargs): # name placeholder ignored
        self.dataset_id = dataset_id
        try:
            self.openml_dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False)
        except Exception as e: print(f"ERROR: OpenML metadata fetch for ID {dataset_id} failed: {e}"); raise
        
        name = self.openml_dataset.name.replace(" ", "_").replace("'", "").lower()
        target_col = kwargs.get("target_col", self.openml_dataset.default_target_attribute)
        super().__init__(name, input_base_dir, target_col=target_col, **kwargs)

    def get_raw_dataset(self):
        print(f"INFO: Downloading OpenML data for '{self.name}' (ID: {self.dataset_id})")
        try:
            df, *_ = self.openml_dataset.get_data(dataset_format="dataframe") # Download now
            # Optionally save raw download to *input* "raw" folder if needed for inspection/cache later
            # (self.input_specific_dataset_path / "raw").mkdir(parents=True, exist_ok=True)
            # df.to_csv(self.input_specific_dataset_path / "raw" / "train_openml_download.csv", index=False)
            return df, None # No separate raw test file from OpenML typically
        except Exception as e: print(f"ERROR: OpenML data download for {self.name} failed: {e}"); traceback.print_exc(); return None, None

    def get_dataset_info(self):
        info = super().get_dataset_info() # Calls this class's get_raw_dataset
        info["description"] = self.openml_dataset.description or info.get("description")
        info["metadata"].update(self.openml_dataset.qualities or {})
        return info

async def process_dataset(ds_obj: ExpDataset, sd_inst: SolutionDesigner, save_pool: bool, ds_yaml_dict: dict):
    print(f"INFO: Async processing for '{ds_obj.name}'. Save analysis pool: {save_pool}")
    if save_pool and sd_inst:
        print(f"INFO: SolutionDesigner generating solutions for '{ds_obj.name}'")
        try: await sd_inst.generate_solutions(ds_obj.get_dataset_info(), ds_obj.name)
        except Exception as e: print(f"ERROR: SolutionDesigner failed for {ds_obj.name}: {e}"); traceback.print_exc()
    try: ds_yaml_dict.setdefault("datasets", {})[ds_obj.name] = create_dataset_dict(ds_obj)
    except Exception as e: print(f"ERROR: create_dataset_dict failed for {ds_obj.name}: {e}"); traceback.print_exc()


def parse_args():
    p = argparse.ArgumentParser(description="SELA Dataset Processor")
    p.add_argument("--dataset", type=str, help="Custom dataset name (folder under input_datasets_dir).")
    p.add_argument("--target_col", type=str, help="Target column for custom dataset.")
    p.add_argument("--openml_id", type=int, help="OpenML dataset ID.")
    p.add_argument("--force_update", action="store_true", help="Force overwrite of existing processed files.")
    p.add_argument("--save_analysis_pool", action="store_true", help="Generate insights with SolutionDesigner.")
    p.add_argument("--no_save_analysis_pool", dest="save_analysis_pool", action="store_false")
    p.set_defaults(save_analysis_pool=False)
    return p.parse_args()

if __name__ == "__main__":
    print("--- Running dataset.py script (SELA v_rely_on_utils_config) ---")
    cli_args = parse_args()

    # DATA_CONFIG is imported from the now-modified utils.py.
    # It should already contain correctly configured:
    # DATA_CONFIG["datasets_dir"] (for input, e.g., /repository/datasets/)
    # DATA_CONFIG["work_dir"] (e.g., /tmp/sela/workspace)
    # DATA_CONFIG["processed_datasets_output_dir"] (e.g., /tmp/sela/workspace/sela_datasets_output)
    # DATA_CONFIG["role_dir"]
    # DATA_CONFIG["datasets"] (metadata from datasets.yaml)
    
    print(f"INFO: dataset.py __main__: Using DATA_CONFIG from utils.py: {json.dumps(DATA_CONFIG, indent=2, default=str)}")

    # This is the base directory for *reading* raw input datasets, taken from the configured DATA_CONFIG
    input_dir_for_constructors = DATA_CONFIG.get("datasets_dir")
    if not input_dir_for_constructors:
        print("CRITICAL ERROR: 'datasets_dir' for input not found in DATA_CONFIG from utils.py. Exiting.")
        sys.exit(1)
    if "processed_datasets_output_dir" not in DATA_CONFIG: # Should be set by utils.py
        print("CRITICAL ERROR: 'processed_datasets_output_dir' not found in DATA_CONFIG from utils.py. Exiting.")
        sys.exit(1)
    if "work_dir" not in DATA_CONFIG: # Should be set by utils.py
        print("CRITICAL ERROR: 'work_dir' not found in DATA_CONFIG from utils.py. Exiting.")
        sys.exit(1)

    force_update_run = cli_args.force_update
    save_analysis_pool_run = cli_args.save_analysis_pool
    
    final_datasets_yaml_data = {"datasets": {}}

    solution_designer_instance = None
    if save_analysis_pool_run:
        try:
            if SolutionDesigner is object: 
                raise NameError("SolutionDesigner was not successfully imported (placeholder 'object' found).")
            solution_designer_instance = SolutionDesigner() 
            print("INFO: SolutionDesigner initialized for saving analysis pool.")
        except NameError as ne: 
            print(f"WARN: SolutionDesigner class not available ({ne}). Cannot save analysis pool.")
            save_analysis_pool_run = False 
        except Exception as e:
            print(f"WARN: Failed to initialize SolutionDesigner instance: {e}. Analysis pool saving will be disabled.")
            traceback.print_exc()
            save_analysis_pool_run = False

    num_datasets_processed = 0
    if cli_args.openml_id:
        print(f"INFO: Processing OpenML ID: {cli_args.openml_id}")
        try:
            ds = OpenMLExpDataset("", input_dir_for_constructors, cli_args.openml_id, 
                                  target_col=cli_args.target_col, force_update=force_update_run)
            asyncio.run(process_dataset(ds, solution_designer_instance, save_analysis_pool_run, final_datasets_yaml_data))
            num_datasets_processed += 1
        except Exception as e: print(f"ERROR: OpenML ID {cli_args.openml_id} failed: {e}"); traceback.print_exc()
    
    elif cli_args.dataset:
        print(f"INFO: Processing custom dataset '{cli_args.dataset}' (target: '{cli_args.target_col}')")
        if not cli_args.target_col: print("ERROR: --target_col required for --dataset."); sys.exit(1)
        try:
            ds = ExpDataset(cli_args.dataset, input_dir_for_constructors, 
                            target_col=cli_args.target_col, force_update=force_update_run)
            asyncio.run(process_dataset(ds, solution_designer_instance, save_analysis_pool_run, final_datasets_yaml_data))
            num_datasets_processed += 1
        except Exception as e: print(f"ERROR: Custom dataset {cli_args.dataset} failed: {e}"); traceback.print_exc()
        
    else: 
        print("INFO: No specific dataset arg. Checking predefined lists (OPENML_DATASET_IDS, CUSTOM_DATASETS).")
        # (Loops for predefined lists - these lists are empty in the current dataset.py)

    if num_datasets_processed == 0: print("WARN: No datasets were processed successfully in this run.")
    else: print(f"INFO: Completed processing for {num_datasets_processed} dataset(s).")

    output_datasets_yaml_file = "datasets.yaml" 
    try:
        from metagpt.ext.sela.utils import DEFAULT_DATASETS_YAML_PATH as sela_yaml_path
        output_datasets_yaml_file = str(sela_yaml_path)
    except ImportError:
        output_datasets_yaml_file = str(Path(DATA_CONFIG.get("work_dir", ".")) / "datasets_processed_manifest.yaml")
        
    save_datasets_dict_to_yaml(final_datasets_yaml_data, name_or_path=output_datasets_yaml_file)
    print(f"--- dataset.py script finished ---")