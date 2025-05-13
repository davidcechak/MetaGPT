# /tmp/MetaGPT_fork_sela/metagpt/ext/sela/utils.py
"""Utility helpers that glue SELA to MetaGPT and keep all path configuration in **one** place.

Key points versus the previous revision
--------------------------------------
* `work_dir` is no longer hard‑wired to ``/tmp/sela/workspace`` – it is resolved in this order:
    1. the **environment variable** ``SELA_WORK_DIR`` if it is set;
    2. the value already present in *data.yaml* (written by ``overwrite_data_yaml.sh``);
    3. the historical fallback ``/tmp/sela/workspace``.
* `processed_datasets_output_dir` is recomputed from `work_dir` so every module reads & writes under the same run folder (e.g. ``/workspace/runs/<run‑id>/sela_datasets_output``).
* A tiny bug‑fix: the missing ``import json`` required by ``clean_json_from_rsp`` is now present.

Everything else is unchanged.
"""

from __future__ import annotations

import os
import re
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import nbformat  # notebook operations
from nbformat.notebooknode import NotebookNode
import yaml
from loguru import logger as _logger

try:
    from metagpt.roles.role import Role
except ImportError:  # when MetaGPT is not installed yet (during image build, for example)
    print("WARN: utils.py: Could not import 'Role' from 'metagpt.roles.role'. Using placeholder.")
    Role = object  # type: ignore

###############################################################################
# Configuration loading helpers                                               #
###############################################################################

_UTILS_DIR = Path(__file__).parent


def _load_yaml(file_name_or_path: str) -> dict:
    """Load a YAML file returning an **empty dict** on any failure."""
    config_path = _UTILS_DIR / file_name_or_path
    if not config_path.exists():
        # Try relative to the current CWD (fallback for unit‑tests)
        config_path = Path(file_name_or_path)
    try:
        with open(config_path, "r") as fh:
            data = yaml.safe_load(fh)
            return data or {}
    except Exception as exc:
        print(f"ERROR: utils.py: Could not load YAML '{config_path}': {exc}")
        traceback.print_exc()
        return {}


# Base configuration comes from the repo‑level YAML files --------------------
DATA_CONFIG: dict = _load_yaml("data.yaml")
DATASET_METADATA_CONFIG: dict = _load_yaml("datasets.yaml")

###############################################################################
# Path resolution                                                             #
###############################################################################

# 1. Where raw, *input* datasets live ----------------------------------------
#    (Only overridden here if absent so that data.yaml or SELA_DATASETS_DIR can win.)
DATA_CONFIG.setdefault("datasets_dir", "/repository/datasets/")

# 2. Where **this run** should store every artefact --------------------------
_env_work_dir = os.getenv("SELA_WORK_DIR")
if _env_work_dir:  # Highest priority: explicit environment variable
    DATA_CONFIG["work_dir"] = _env_work_dir
else:  # Else respect whatever overwrite_data_yaml.sh placed in data.yaml, or fall back.
    DATA_CONFIG.setdefault("work_dir", "/tmp/sela/workspace")

# 3. Derived: where processed splits, info.json, etc. should be written ------
DATA_CONFIG["processed_datasets_output_dir"] = str(
    Path(DATA_CONFIG["work_dir"]) / "sela_datasets_output"
)

# 4. Other misc defaults ------------------------------------------------------
DATA_CONFIG.setdefault("role_dir", "storage/SELA")
# Merge dataset‑specific metadata
DATA_CONFIG["datasets"] = DATASET_METADATA_CONFIG.get("datasets", {})

DEFAULT_DATASETS_YAML_PATH = _UTILS_DIR / "datasets.yaml"  # for code that appends

print(
    "INFO: utils.py: DATA_CONFIG initialised →\n" + yaml.dump(DATA_CONFIG, default_flow_style=False, sort_keys=False)
)

###############################################################################
# Logging helper                                                              #
###############################################################################

def _configure_logger() -> _logger.__class__:
    logfile_level = DATA_CONFIG.get("logfile_level", "DEBUG")
    today = datetime.now().strftime("%Y%m%d")
    log_name = f"{today}"

    try:
        _logger.remove()
    except ValueError:
        pass  # no handler yet

    _logger.level("MCTS", color="<green>", no=25)

    log_dir = Path(DATA_CONFIG["work_dir"]) / DATA_CONFIG["role_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger.add(sys.stderr, level="INFO")
    _logger.add(log_dir / f"{log_name}.txt", level=logfile_level, rotation="10 MB", retention="7 days")
    _logger.propagate = False
    return _logger


mcts_logger = _configure_logger()

###############################################################################
# Notebook utilities                                                          #
###############################################################################

def get_exp_pool_path(task_name: str, data_config_param: dict, pool_name: str = "analysis_pool") -> str | None:
    """Return the full path to *analysis_pool.json* for a given dataset if present."""
    datasets_dir = data_config_param.get("datasets_dir")
    if not datasets_dir:
        raise ValueError("'datasets_dir' not provided in data_config_param")
    candidate = Path(datasets_dir) / task_name / f"{pool_name}.json"
    print(f"INFO: Looking for {pool_name}.json at: {candidate}")
    return str(candidate) if candidate.exists() else None


def change_plan(role: Role, plan: str) -> bool:
    """Swap the *plan* of the **first unfinished** task. Return *True* if all tasks were already finished."""
    print(f"Change next plan to: {plan}")
    if not (
        hasattr(role, "planner") and hasattr(role.planner, "plan") and hasattr(role.planner.plan, "tasks")
    ):
        print("WARN: change_plan – Role, planner, or tasks missing → cannot update plan.")
        return False

    tasks = role.planner.plan.tasks
    for idx, task in enumerate(tasks):
        if not task.code:
            tasks[idx].plan = plan  # first unfinished
            return False
    return True  # all tasks finished


def _is_cell_to_delete(cell: NotebookNode) -> bool:
    if cell.get("outputs"):
        return any("traceback" in output for output in cell["outputs"] if output)
    return False


def _process_cells(nb: NotebookNode) -> NotebookNode:
    """Strip traceback cells and renumber execution_count sequentially."""
    new_cells, exec_count = [], 1
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if not _is_cell_to_delete(cell):
                cell["execution_count"] = exec_count
                exec_count += 1
                new_cells.append(cell)
        else:
            new_cells.append(cell)
    nb["cells"] = new_cells
    return nb


def save_notebook(role: Role, save_dir: str = "", name: str = "", save_to_depth: bool = False) -> None:
    base = Path(save_dir) if save_dir else Path(DATA_CONFIG["work_dir"]) / "notebook_outputs"
    base.mkdir(parents=True, exist_ok=True)

    if not (hasattr(role, "execute_code") and hasattr(role.execute_code, "nb")):
        print("WARN: save_notebook – role.execute_code.nb missing.")
        return

    cleaned = _process_cells(role.execute_code.nb)
    fname = name or f"notebook_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    nbformat.write(cleaned, base / f"{fname}.ipynb")
    print(f"INFO: Notebook saved to {base/f'{fname}.ipynb'}")

    if save_to_depth and hasattr(role, "planner"):
        codes = [t.code for t in role.planner.plan.tasks if t.code]
        depth_nb = nbformat.v4.new_notebook()
        depth_nb.cells = [nbformat.v4.new_code_cell(c) for c in codes]
        nbformat.write(depth_nb, base / f"{fname}_clean.ipynb")
        print(f"INFO: Cleaned notebook saved to {base/f'{fname}_clean.ipynb'}")

###############################################################################
# Misc helpers                                                                #
###############################################################################

def clean_json_from_rsp(text: str) -> str:
    """Extract a JSON block from a model response wrapped in ```json ... ``` fences."""
    if not isinstance(text, str):
        return ""

    fenced = re.findall(r"```json(.*?)```", text, re.DOTALL)
    if fenced:
        json_str = "".join(fenced).strip()
    else:
        json_str = text.strip()

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        return ""

print("INFO: utils.py loaded – unified DATA_CONFIG ready for SELA.")
