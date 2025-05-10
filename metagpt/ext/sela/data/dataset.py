import argparse
import asyncio
import json
import os
from pathlib import Path
import sys # For sys.exit
import traceback # For error printing, if needed

import openml
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Try to import from SELA extension, with fallbacks if not found during standalone execution
try:
    from metagpt.ext.sela.insights.solution_designer import SolutionDesigner
    # Import DATA_CONFIG and other constants from the utils module
    # Ensure utils.py is also in a state where DATA_CONFIG is loaded correctly or hardcoded.
    from metagpt.ext.sela.utils import DATA_CONFIG, SPECIAL_INSTRUCTIONS, DI_INSTRUCTION, TASK_PROMPT
except ImportError as e:
    print(f"WARN: dataset.py: Could not import SELA modules: {e}. Using placeholders.")
    SolutionDesigner = object # Placeholder
    DATA_CONFIG = {}        # Placeholder, will be populated in __main__
    # Define constants locally if import fails (ensure these match your intended values)
    SPECIAL_INSTRUCTIONS = {"ag": "- Please use autogluon...", "stacking": "- To avoid overfitting...", "text": "- You could use models from transformers...", "image": "- You could use models from transformers/torchvision..."} # Truncated for brevity
    DI_INSTRUCTION = """## Attention
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
    TASK_PROMPT = """# User requirement
{user_requirement}
{additional_instruction}
# Data dir
train set (with labels): {train_path}
dev set (with labels): {dev_path}
test set (without labels): {test_path}
dataset description: {data_info_path} (During EDA, you can use this file to get additional information about the dataset)
"""


BASE_USER_REQUIREMENT = """
This is a {datasetname} dataset. Your goal is to predict the target column `{target_col}`.
Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. 
Report {metric} on the eval data. Do not plot or make any visualizations.
"""

# USE_AG, TEXT_MODALITY, IMAGE_MODALITY, STACKING can be fully defined here or imported if utils.py is reliable

SEED = 100
TRAIN_TEST_SPLIT = 0.8
TRAIN_DEV_SPLIT = 0.75

OPENML_DATASET_IDS = [
]

# CUSTOM_DATASETS will be populated from args in __main__
CUSTOM_DATASETS = []

DSAGENT_DATASETS = [
]


def get_split_dataset_path(dataset_name, config): # config here is expected to be DATA_CONFIG
    datasets_dir = config.get("datasets_dir") # Use .get() for safety
    if not datasets_dir:
        print(f"ERROR: get_split_dataset_path: 'datasets_dir' not found in config: {config}")
        # Fallback or raise error - for now, let it proceed and potentially fail if path is essential
        # Or use a default if appropriate: datasets_dir = "/repository/datasets/"
        raise ValueError("datasets_dir missing in config for get_split_dataset_path")

    if dataset_name in config.get("datasets", {}):
        dataset = config["datasets"][dataset_name]
        data_path = os.path.join(datasets_dir, dataset["dataset"]) # dataset["dataset"] should be dataset_name
        split_datasets = {
            "train": os.path.join(data_path, "split_train.csv"),
            "dev": os.path.join(data_path, "split_dev.csv"),
            "dev_wo_target": os.path.join(data_path, "split_dev_wo_target.csv"),
            "dev_target": os.path.join(data_path, "split_dev_target.csv"),
            "test": os.path.join(data_path, "split_test.csv"), # This might be test_with_target if generated
            "test_wo_target": os.path.join(data_path, "split_test_wo_target.csv"),
            "test_target": os.path.join(data_path, "split_test_target.csv"),
        }
        return split_datasets
    else:
        # Fallback for datasets not explicitly in config["datasets"] but for which paths might be derivable
        # This could happen if dataset.py is run standalone for a new dataset.
        data_path = os.path.join(datasets_dir, dataset_name)
        print(f"WARN: Dataset {dataset_name} not in config file's 'datasets' list. Assuming direct path structure under {datasets_dir}.")
        split_datasets = {
            "train": os.path.join(data_path, "split_train.csv"),
            "dev": os.path.join(data_path, "split_dev.csv"),
            "dev_wo_target": os.path.join(data_path, "split_dev_wo_target.csv"),
            "dev_target": os.path.join(data_path, "split_dev_target.csv"),
            "test": os.path.join(data_path, "split_test.csv"),
            "test_wo_target": os.path.join(data_path, "split_test_wo_target.csv"),
            "test_target": os.path.join(data_path, "split_test_target.csv"),
        }
        return split_datasets
        # Original error:
        # raise ValueError(
        #     f"Dataset {dataset_name} not found in config file. Available datasets: {config.get('datasets', {}).keys()}"
        # )

def get_user_requirement(task_name, config): # config here is expected to be DATA_CONFIG
    if task_name in config.get("datasets", {}):
        dataset_meta = config["datasets"][task_name]
        user_requirement = dataset_meta.get("user_requirement")
        if not user_requirement:
             print(f"WARN: 'user_requirement' not found for {task_name} in datasets.yaml. Generating default.")
             # Fallback to generate base requirement if not in datasets.yaml
             metric = dataset_meta.get("metric", "relevant metric (e.g. f1_binary or rmse)") # Provide a sensible default
             target_col = dataset_meta.get("target_col", "target") # Provide a sensible default
             return BASE_USER_REQUIREMENT.format(datasetname=task_name, target_col=target_col, metric=metric)
        return user_requirement
    else:
        print(f"WARN: Dataset {task_name} not explicitly in datasets.yaml. Generating default user_requirement.")
        # Fallback to generate base requirement
        # This requires knowing target_col and metric, which might not be available here easily
        # For now, provide a generic one. ExpDataset.create_base_requirement might be better if an instance exists.
        return BASE_USER_REQUIREMENT.format(datasetname=task_name, target_col="target_col_placeholder", metric="metric_placeholder")
        # Original error:
        # raise ValueError(
        #     f"Dataset {task_name} not found in config file. Available datasets: {config.get('datasets', {}).keys()}"
        # )


def save_datasets_dict_to_yaml(datasets_dict, name="datasets.yaml"):
    # This should ideally save to a path derived from DATA_CONFIG["work_dir"] or similar,
    # not just the current working directory, unless that's intended.
    # For now, keeping original behavior.
    target_path = Path(name)
    if DATA_CONFIG and DATA_CONFIG.get("work_dir"): # Save inside work_dir if possible
        target_path = Path(DATA_CONFIG["work_dir"]) / name
        target_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    print(f"INFO: Saving datasets dictionary to {target_path}")
    with open(target_path, "w") as file:
        yaml.dump(datasets_dict, file, default_flow_style=False, sort_keys=False)


def create_dataset_dict(dataset: 'ExpDataset'): # Forward reference ExpDataset
    dataset_dict = {
        "dataset": dataset.name, # This should be just the name, not the full path.
        "user_requirement": dataset.create_base_requirement(),
        "metric": dataset.get_metric(),
        "target_col": dataset.target_col,
    }
    return dataset_dict


def generate_di_instruction(output_dir, special_instruction_key):
    special_instruction_prompt = ""
    if special_instruction_key and special_instruction_key in SPECIAL_INSTRUCTIONS:
        special_instruction_prompt = SPECIAL_INSTRUCTIONS[special_instruction_key]
    elif special_instruction_key:
        print(f"WARN: Special instruction key '{special_instruction_key}' not found in SPECIAL_INSTRUCTIONS.")

    additional_instruction = DI_INSTRUCTION.format(
        output_dir=output_dir, special_instruction=special_instruction_prompt
    )
    return additional_instruction


def generate_task_requirement(task_name, data_config_arg, is_di=True, special_instruction_key=None):
    user_requirement = get_user_requirement(task_name, data_config_arg) # data_config_arg should be DATA_CONFIG
    split_dataset_path = get_split_dataset_path(task_name, data_config_arg)

    train_path = split_dataset_path["train"]
    dev_path = split_dataset_path["dev"]
    test_path = split_dataset_path["test_wo_target"] # This is the test set *without* labels for inference

    work_dir = data_config_arg.get("work_dir")
    if not work_dir:
        print("ERROR: generate_task_requirement: 'work_dir' not in data_config. Using '.'")
        work_dir = "." # Fallback to current directory
    output_dir = Path(work_dir) / task_name # Use Path object for robust path joining

    datasets_dir = data_config_arg.get("datasets_dir")
    if not datasets_dir:
        print("ERROR: generate_task_requirement: 'datasets_dir' not in data_config. Data info path may be incorrect.")
        data_info_path = Path(task_name) / "dataset_info.json" # Fallback
    else:
        data_info_path = Path(datasets_dir) / task_name / "dataset_info.json"

    additional_instruction = ""
    if is_di:
        additional_instruction = generate_di_instruction(str(output_dir), special_instruction_key) # Ensure output_dir is string

    # Ensure paths are strings for formatting
    task_prompt_formatted = TASK_PROMPT.format(
        user_requirement=user_requirement,
        train_path=str(train_path),
        dev_path=str(dev_path),
        test_path=str(test_path),
        additional_instruction=additional_instruction,
        data_info_path=str(data_info_path),
    )
    print(f"INFO: Generated task requirement for '{task_name}':\n{task_prompt_formatted}")
    return task_prompt_formatted


class ExpDataset:
    description: str = None
    metadata: dict = None
    dataset_dir: str = None # This will be the base datasets directory (e.g., /repository/datasets/)
    target_col: str = None
    name: str = None       # This will be the dataset name (e.g., human_nontata_promoters)

    def __init__(self, name, dataset_dir_base, **kwargs): # dataset_dir_base is like /repository/datasets/
        self.name = name
        self.dataset_dir = dataset_dir_base # Store the base datasets path
        self.target_col = kwargs.get("target_col", "target") # Default target column name
        self.force_update = kwargs.get("force_update", False)
        
        # Construct the full path to the specific dataset's directory
        self.specific_dataset_path = Path(self.dataset_dir) / self.name
        self.specific_dataset_path.mkdir(parents=True, exist_ok=True) # Ensure it exists

        print(f"INFO: Initializing ExpDataset '{self.name}' with base directory '{self.dataset_dir}', specific path '{self.specific_dataset_path}'")
        self.save_dataset(target_col=self.target_col)

    def check_dataset_exists(self):
        # Checks for split files in the specific dataset's directory
        fnames = [
            "split_train.csv", "split_dev.csv", "split_test.csv",
            "split_dev_wo_target.csv", "split_dev_target.csv",
            "split_test_wo_target.csv", "split_test_target.csv",
        ]
        for fname in fnames:
            if not (self.specific_dataset_path / fname).exists():
                return False
        return True

    def check_datasetinfo_exists(self):
        return (self.specific_dataset_path / "dataset_info.json").exists()

    def get_raw_dataset(self):
        # MODIFIED: Looks for train.csv directly in dataset_base_path first, then falls back to 'raw' subdir
        dataset_base_path = self.specific_dataset_path # e.g., /repository/datasets/human_nontata_promoters
        
        direct_train_csv_path = Path(dataset_base_path, "train.csv")
        direct_test_csv_path = Path(dataset_base_path, "test.csv")

        raw_subdir_path = Path(dataset_base_path, "raw")
        original_style_train_csv_path = Path(raw_subdir_path, "train.csv")
        original_style_test_csv_path = Path(raw_subdir_path, "test.csv")

        train_df = None
        test_df = None

        print(f"INFO: ExpDataset.get_raw_dataset: Attempting to load train data for '{self.name}' from base '{dataset_base_path}'")

        if os.path.exists(direct_train_csv_path):
            print(f"INFO: Found train.csv at direct path: {direct_train_csv_path}")
            train_df = pd.read_csv(direct_train_csv_path)
            if os.path.exists(direct_test_csv_path):
                print(f"INFO: Found test.csv at direct path: {direct_test_csv_path}")
                test_df = pd.read_csv(direct_test_csv_path)
        elif os.path.exists(original_style_train_csv_path):
            print(f"INFO: Found train.csv at original 'raw' subdirectory path: {original_style_train_csv_path}")
            train_df = pd.read_csv(original_style_train_csv_path)
            if os.path.exists(original_style_test_csv_path):
                print(f"INFO: Found test.csv at original 'raw' subdirectory path: {original_style_test_csv_path}")
                test_df = pd.read_csv(original_style_test_csv_path)
        else:
            raise FileNotFoundError(
                f"Raw dataset `train.csv` not found in {dataset_base_path} OR {raw_subdir_path} for dataset '{self.name}'"
            )
            
        return train_df, test_df

    def get_dataset_info(self):
        # Uses the specific_dataset_path to potentially find a raw train.csv for info
        # This assumes get_raw_dataset will successfully return train_df if it exists
        train_df_for_info, _ = self.get_raw_dataset() # Get train_df to derive info

        if train_df_for_info is None:
            raise RuntimeError(f"Cannot get dataset info for {self.name} as raw train.csv could not be loaded.")

        if not self.target_col or self.target_col not in train_df_for_info.columns:
             print(f"WARN: Target column '{self.target_col}' not found in DataFrame for {self.name}. Attempting to infer or using placeholder.")
             # Attempt to infer target or use a default, this part needs robust handling
             # For now, if target_col is critical and missing, this will error out later or give bad info.
             # A simple placeholder:
             effective_target_col = train_df_for_info.columns[-1] if not train_df_for_info.empty else "unknown_target"
             if self.target_col and self.target_col not in train_df_for_info.columns:
                 print(f"WARN: Specified target_col '{self.target_col}' not in columns. Using inferred '{effective_target_col}'.")
             self.target_col = effective_target_col # Update if inferred

        metadata = {
            "NumberOfClasses": train_df_for_info[self.target_col].nunique() if self.target_col in train_df_for_info else 0,
            "NumberOfFeatures": train_df_for_info.shape[1],
            "NumberOfInstances": train_df_for_info.shape[0],
            "NumberOfInstancesWithMissingValues": int(train_df_for_info.isnull().any(axis=1).sum()),
            "NumberOfMissingValues": int(train_df_for_info.isnull().sum().sum()),
            "NumberOfNumericFeatures": train_df_for_info.select_dtypes(include=["number"]).shape[1],
            "NumberOfSymbolicFeatures": train_df_for_info.select_dtypes(include=["object"]).shape[1],
        }
        df_head_text = self.get_df_head(train_df_for_info)
        dataset_info_dict = {
            "name": self.name,
            "description": self.description or f"Dataset information for {self.name}", # Add default description
            "target_col": self.target_col,
            "metadata": metadata,
            "df_head": df_head_text,
        }
        return dataset_info_dict

    def get_df_head(self, df_to_head):
        return df_to_head.head().to_string(index=False)

    def get_metric(self):
        dataset_info = self.get_dataset_info()
        num_classes = dataset_info["metadata"]["NumberOfClasses"]
        if num_classes == 2:
            metric = "f1_binary" # Consistent naming
        elif 2 < num_classes <= 200:
            metric = "f1_weighted"
        elif num_classes > 200 or num_classes == 0: # num_classes == 0 for regression
            metric = "rmse"
        else: # Should not happen if target_col logic is fine
            print(f"WARN: Unexpected number of classes {num_classes} for {self.name}. Defaulting metric to rmse.")
            metric = "rmse"
        return metric

    def create_base_requirement(self):
        metric = self.get_metric()
        req = BASE_USER_REQUIREMENT.format(datasetname=self.name, target_col=self.target_col, metric=metric)
        return req

    def save_dataset(self, target_col): # target_col is passed but self.target_col should be set in __init__
        if not self.target_col: # Ensure self.target_col is set
            self.target_col = target_col if target_col else "target" # Fallback
            print(f"WARN: self.target_col was not set, using '{self.target_col}' for {self.name}")

        df, test_df_raw = self.get_raw_dataset() # df is train_df from raw
        if df is None:
            print(f"ERROR: Could not load raw training data for {self.name}. Aborting save_dataset.")
            return

        if not self.check_dataset_exists() or self.force_update:
            print(f"INFO: Splitting and saving dataset '{self.name}' in '{self.specific_dataset_path}'")
            self.split_and_save(df, self.target_col, test_df_raw=test_df_raw) # Use self.target_col
        else:
            print(f"INFO: Split datasets for '{self.name}' already exist at '{self.specific_dataset_path}'. Skipping save.")

        if not self.check_datasetinfo_exists() or self.force_update:
            print(f"INFO: Saving dataset_info.json for '{self.name}' at '{self.specific_dataset_path}'")
            try:
                dataset_info = self.get_dataset_info()
                self.save_datasetinfo(dataset_info)
            except Exception as e:
                print(f"ERROR: Failed to get or save dataset_info for {self.name}: {e}")
                traceback.print_exc()
        else:
            print(f"INFO: dataset_info.json for '{self.name}' already exists. Skipping save.")


    def save_datasetinfo(self, dataset_info_dict):
        info_path = self.specific_dataset_path / "dataset_info.json"
        try:
            with open(info_path, "w", encoding="utf-8") as file:
                json.dump(dataset_info_dict, file, indent=4, ensure_ascii=False)
            print(f"INFO: Successfully saved dataset_info.json to {info_path}")
        except Exception as e:
            print(f"ERROR: Could not write dataset_info.json to {info_path}: {e}")

    def save_split_datasets(self, df_to_save, split_name, target_col_to_use):
        # Saves to self.specific_dataset_path
        path = self.specific_dataset_path
        output_path_main = path / f"split_{split_name}.csv"
        df_to_save.to_csv(output_path_main, index=False)
        print(f"INFO: Saved {output_path_main}")

        if target_col_to_use and target_col_to_use in df_to_save.columns:
            df_wo_target = df_to_save.drop(columns=[target_col_to_use])
            output_path_wo_target = path / f"split_{split_name}_wo_target.csv"
            df_wo_target.to_csv(output_path_wo_target, index=False)
            print(f"INFO: Saved {output_path_wo_target}")
            
            df_target_only = df_to_save[[target_col_to_use]].copy()
            # Standardize target column name to 'target' in this file
            if target_col_to_use != "target":
                df_target_only["target"] = df_target_only[target_col_to_use]
                df_target_only = df_target_only.drop(columns=[target_col_to_use])
            output_path_target_only = path / f"split_{split_name}_target.csv"
            df_target_only.to_csv(output_path_target_only, index=False)
            print(f"INFO: Saved {output_path_target_only}")
        elif target_col_to_use:
             print(f"WARN: Target column '{target_col_to_use}' not found in DataFrame for split '{split_name}'. Cannot save _wo_target or _target.csv versions.")


    def split_and_save(self, train_df_raw, target_col_to_use, test_df_raw=None):
        if not target_col_to_use:
            raise ValueError("Target column not provided for split_and_save")
        if target_col_to_use not in train_df_raw.columns:
            raise ValueError(f"Target column '{target_col_to_use}' not found in training data columns: {train_df_raw.columns}")

        if test_df_raw is None: # If no separate raw test file, split train_df_raw
            print(f"INFO: No raw test_df provided for {self.name}. Splitting train_df_raw for train/test sets.")
            train_set, test_set_with_target = train_test_split(train_df_raw, test_size=1 - TRAIN_TEST_SPLIT, random_state=SEED, stratify=train_df_raw[target_col_to_use] if train_df_raw[target_col_to_use].nunique() > 1 else None)
        else: # Raw test file is provided
            print(f"INFO: Using provided raw test_df for {self.name}.")
            train_set = train_df_raw
            test_set_with_target = test_df_raw
            if target_col_to_use not in test_set_with_target.columns:
                 print(f"WARN: Target column '{target_col_to_use}' not found in provided test_df_raw. Test target files will be empty or error.")
                 # Create an empty target column or handle as appropriate
                 test_set_with_target[target_col_to_use] = pd.NA 


        # Split train_set further into train and dev
        train_final, dev_set_with_target = train_test_split(train_set, test_size=1 - TRAIN_DEV_SPLIT, random_state=SEED, stratify=train_set[target_col_to_use] if train_set[target_col_to_use].nunique() > 1 else None)

        self.save_split_datasets(train_final, "train", target_col_to_use) # Train set always has target
        self.save_split_datasets(dev_set_with_target, "dev", target_col_to_use)
        self.save_split_datasets(test_set_with_target, "test", target_col_to_use)


class OpenMLExpDataset(ExpDataset):
    def __init__(self, name_ignored, dataset_dir_base, dataset_id, **kwargs): # name is derived from OpenML
        self.dataset_id = dataset_id
        try:
            self.dataset = openml.datasets.get_dataset(
                self.dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False # Set to False initially
            )
        except Exception as e:
            print(f"ERROR: Failed to get OpenML dataset metadata for ID {self.dataset_id}: {e}")
            raise
            
        openml_name = self.dataset.name.replace(" ", "_").replace("'", "").lower() # Sanitize name
        print(f"INFO: OpenML dataset ID {self.dataset_id} resolved to name '{openml_name}'")
        # Default target attribute from OpenML, can be overridden by kwargs
        target_col_openml = self.dataset.default_target_attribute
        target_col_kwarg = kwargs.get("target_col", None)
        final_target_col = target_col_kwarg if target_col_kwarg else target_col_openml

        if not final_target_col:
            print(f"WARN: No default_target_attribute for OpenML ID {self.dataset_id} and no target_col in kwargs. This might lead to issues.")
            # Attempt to use last column or handle error
            # For now, will proceed and might fail in get_dataset_info or metric calculation

        super().__init__(openml_name, dataset_dir_base, target_col=final_target_col, **kwargs)

    def get_raw_dataset(self):
        # Download data only when this method is called
        print(f"INFO: Downloading data for OpenML dataset '{self.name}' (ID: {self.dataset_id})")
        try:
            dataset_df, *_ = self.dataset.get_data(dataset_format="dataframe") # Ensure it's pandas DataFrame
        except Exception as e:
            print(f"ERROR: Failed to download data for OpenML dataset {self.name} (ID: {self.dataset_id}): {e}")
            return None, None # Return Nones if download fails

        # Save the raw downloaded data to the 'raw' subdirectory for consistency or inspection
        # raw_files_dir = self.specific_dataset_path / "raw" # Use specific_dataset_path
        # raw_files_dir.mkdir(parents=True, exist_ok=True)
        # raw_train_path = raw_files_dir / "train.csv"
        # dataset_df.to_csv(raw_train_path, index=False)
        # print(f"INFO: Saved raw OpenML data to {raw_train_path}")
        return dataset_df, None # OpenML usually provides one dataset, treat as train, test_df is None

    def get_dataset_info(self):
        dataset_info_dict = super().get_dataset_info() # Calls get_raw_dataset internally
        # Update with OpenML specific metadata if available
        dataset_info_dict["description"] = self.dataset.description or dataset_info_dict.get("description")
        dataset_info_dict["metadata"].update(self.dataset.qualities or {})
        return dataset_info_dict


async def process_dataset(dataset_obj: ExpDataset, sol_designer: SolutionDesigner, save_pool: bool, datasets_yaml_dict: dict):
    print(f"INFO: Async process_dataset called for '{dataset_obj.name}'")
    if save_pool and sol_designer: # Check if sol_designer is not None
        print(f"INFO: Generating solutions for '{dataset_obj.name}' (save_analysis_pool is True)")
        try:
            # Ensure dataset_info is correctly generated before passing
            ds_info = dataset_obj.get_dataset_info()
            await sol_designer.generate_solutions(ds_info, dataset_obj.name)
        except Exception as e:
            print(f"ERROR: Failed during solution generation for {dataset_obj.name}: {e}")
            traceback.print_exc()
    else:
        print(f"INFO: Skipping solution generation for '{dataset_obj.name}' (save_analysis_pool is {save_pool} or SolutionDesigner not available)")

    try:
        dataset_dict_entry = create_dataset_dict(dataset_obj)
        datasets_yaml_dict.setdefault("datasets", {})[dataset_obj.name] = dataset_dict_entry
        print(f"INFO: Added '{dataset_obj.name}' to datasets_yaml_dict")
    except Exception as e:
        print(f"ERROR: Failed to create_dataset_dict for {dataset_obj.name}: {e}")
        traceback.print_exc()


def parse_args(): # Renamed from parse_cli_args to match original usage
    parser = argparse.ArgumentParser(description="Process and prepare datasets for SELA.")
    parser.add_argument("--force_update", action="store_true", help="Force update datasets and their info.")
    parser.add_argument("--save_analysis_pool", action="store_true", help="Generate and save analysis pool using SolutionDesigner.")
    parser.add_argument("--no_save_analysis_pool", dest="save_analysis_pool", action="store_false", help="Do not save analysis pool.")
    parser.set_defaults(save_analysis_pool=False) # Default to False as it can be time-consuming

    parser.add_argument("--dataset", type=str, help="Name of the custom dataset to process (folder name).")
    parser.add_argument("--target_col", type=str, help="Target column name for the custom dataset.")
    parser.add_argument("--openml_id", type=int, help="OpenML dataset ID to process.")
    
    return parser.parse_args()


if __name__ == "__main__":
    print("--- Running dataset.py script ---")
    args = parse_args()

    hardcoded_datasets_dir = "/repository/datasets/"
    print(f"INFO: SELA dataset.py: Forcing datasets_dir to: {hardcoded_datasets_dir}")
    datasets_dir = hardcoded_datasets_dir

    hardcoded_work_dir = "/tmp/sela/workspace" 
    hardcoded_role_dir = "storage/SELA"

    if isinstance(DATA_CONFIG, dict):
        DATA_CONFIG["datasets_dir"] = hardcoded_datasets_dir
        if "work_dir" not in DATA_CONFIG or not DATA_CONFIG["work_dir"]:
            DATA_CONFIG["work_dir"] = hardcoded_work_dir
        if "role_dir" not in DATA_CONFIG or not DATA_CONFIG["role_dir"]:
            DATA_CONFIG["role_dir"] = hardcoded_role_dir
    else:
        print(f"WARN: SELA dataset.py: DATA_CONFIG was not a dict. Initializing with hardcoded paths.")
        DATA_CONFIG = {
            "datasets_dir": hardcoded_datasets_dir,
            "work_dir": hardcoded_work_dir,
            "role_dir": hardcoded_role_dir
        }
    
    print(f"INFO: SELA dataset.py: Effective DATA_CONFIG for this run: {DATA_CONFIG}")
    assert DATA_CONFIG["datasets_dir"] == hardcoded_datasets_dir, "Critical: datasets_dir in DATA_CONFIG not correctly set!"

    # Ensure datasets_dir from args or DATA_CONFIG is used by ExpDataset/OpenMLExpDataset
    # The 'datasets_dir' variable is now correctly hardcoded for this script's execution.

    force_update = args.force_update
    save_analysis_pool = args.save_analysis_pool
    
    datasets_yaml_content = {"datasets": {}} # This will be populated and then saved

    # Initialize SolutionDesigner only if needed and possible
    solution_designer_instance = None
    if save_analysis_pool:
        try:
            solution_designer_instance = SolutionDesigner() # Requires DATA_CONFIG
            print("INFO: SolutionDesigner initialized for saving analysis pool.")
        except Exception as e:
            print(f"WARN: Could not initialize SolutionDesigner (needed for save_analysis_pool): {e}. Analysis pool saving will be skipped.")
            save_analysis_pool = False # Disable if it cannot be initialized

    processed_something = False

    # Option 1: Process a specific OpenML dataset if --openml_id is given
    if args.openml_id:
        print(f"INFO: Processing specific OpenML dataset ID: {args.openml_id}")
        try:
            # target_col can also be passed to OpenMLExpDataset via kwargs if needed
            openml_ds_obj = OpenMLExpDataset("", datasets_dir, args.openml_id, target_col=args.target_col, force_update=force_update)
            asyncio.run(process_dataset(openml_ds_obj, solution_designer_instance, save_analysis_pool, datasets_yaml_content))
            processed_something = True
            print(f"INFO: Completed processing OpenML ID: {args.openml_id}")
        except Exception as e:
            print(f"ERROR: Failed processing OpenML ID {args.openml_id}: {e}")
            traceback.print_exc()

    # Option 2: Process a specific custom dataset if --dataset is given
    elif args.dataset:
        print(f"INFO: Processing specific custom dataset: '{args.dataset}' with target column: '{args.target_col}'")
        if not args.target_col:
            print(f"ERROR: Custom dataset '{args.dataset}' specified but --target_col is missing. Cannot proceed.")
            sys.exit(1)
        try:
            # CUSTOM_DATASETS list is not directly used here; args are used instead.
            custom_ds_obj = ExpDataset(args.dataset, datasets_dir, target_col=args.target_col, force_update=force_update)
            asyncio.run(process_dataset(custom_ds_obj, solution_designer_instance, save_analysis_pool, datasets_yaml_content))
            processed_something = True
            print(f"INFO: Completed processing custom dataset: {args.dataset}")
        except Exception as e:
            print(f"ERROR: Failed processing custom dataset {args.dataset}: {e}")
            traceback.print_exc()
            
    # Option 3: (Original behavior) Loop through predefined lists if no specific dataset is given via args
    # This part is now conditional on args.dataset and args.openml_id NOT being provided.
    else:
        print("INFO: No specific dataset via --openml_id or --dataset. Processing predefined lists if populated.")
        # Process OPENML_DATASET_IDS
        for ds_id in OPENML_DATASET_IDS: # This list is empty by default now
            print(f"INFO: Processing OpenML ID from predefined list: {ds_id}")
            try:
                # Assuming target_col needs to be known or is handled by OpenMLExpDataset
                openml_ds_obj = OpenMLExpDataset("", datasets_dir, ds_id, force_update=force_update)
                asyncio.run(process_dataset(openml_ds_obj, solution_designer_instance, save_analysis_pool, datasets_yaml_content))
                processed_something = True
            except Exception as e:
                print(f"ERROR: Failed processing OpenML ID {ds_id} from list: {e}")
                traceback.print_exc()

        # Process CUSTOM_DATASETS (this global list needs to be populated if used this way)
        # Note: The current script populates CUSTOM_DATASETS from args.dataset earlier,
        # so this loop might be redundant or needs CUSTOM_DATASETS to be defined differently for batch mode.
        # For clarity, if args.dataset was given, it's already processed.
        # This loop is for a scenario where CUSTOM_DATASETS is predefined with multiple entries.
        # Example: CUSTOM_DATASETS = [("mydata1", "target1"), ("mydata2", "target2")]
        if not args.dataset: # Only run this loop if not already processed via --dataset arg
            for ds_name, tc in CUSTOM_DATASETS: # This list is likely empty if only using args
                print(f"INFO: Processing custom dataset from predefined list: {ds_name}")
                try:
                    custom_ds_obj = ExpDataset(ds_name, datasets_dir, target_col=tc, force_update=force_update)
                    asyncio.run(process_dataset(custom_ds_obj, solution_designer_instance, save_analysis_pool, datasets_yaml_content))
                    processed_something = True
                except Exception as e:
                    print(f"ERROR: Failed processing custom dataset {ds_name} from list: {e}")
                    traceback.print_exc()
        
        # Process DSAGENT_DATASETS (this list is empty by default)
        for ds_name, tc in DSAGENT_DATASETS:
            print(f"INFO: Processing DSAGENT_DATASET from predefined list: {ds_name}")
            try:
                custom_ds_obj = ExpDataset(ds_name, datasets_dir, target_col=tc, force_update=force_update)
                asyncio.run(process_dataset(custom_ds_obj, solution_designer_instance, save_analysis_pool, datasets_yaml_content))
                processed_something = True
            except Exception as e:
                print(f"ERROR: Failed processing DSAGENT_DATASET {ds_name} from list: {e}")
                traceback.print_exc()

    if not processed_something:
        print("WARN: No datasets were processed. Check arguments or predefined lists.")

    # Save the populated datasets_yaml_content to datasets.yaml
    # This should ideally be the datasets.yaml in metagpt/ext/sela/ not the current dir,
    # or a path specified by DATA_CONFIG.
    # For now, using the default name, which means it saves to current working dir or DATA_CONFIG["work_dir"]
    default_datasets_yaml_path = "datasets.yaml"
    if DATA_CONFIG and DATA_CONFIG.get("work_dir"):
         # Attempt to save to the standard location if utils.py was imported and configured
        try:
            from metagpt.ext.sela.utils import DEFAULT_DATASETS_YAML_PATH as sela_datasets_yaml_path
            default_datasets_yaml_path = sela_datasets_yaml_path
            print(f"INFO: Using SELA default path for datasets.yaml: {default_datasets_yaml_path}")
        except ImportError:
            print(f"WARN: Could not import DEFAULT_DATASETS_YAML_PATH from utils. Saving datasets.yaml to work_dir or CWD.")
            pass # Path will be relative to work_dir or CWD

    save_datasets_dict_to_yaml(datasets_yaml_content, name=str(default_datasets_yaml_path))
    print(f"--- dataset.py script finished. Processed something: {processed_something} ---")