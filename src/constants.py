""""
Author: Md Mostafizur Rahman
File: Configaration file
"""

import os
from pathlib import Path
from typing import List

from typing_extensions import Literal

# directory-related
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
SRC = ROOT / "src"
OUT = ROOT / "output"
MODEL_HT_PARAMS = ROOT / "model_ht_params"
PLOT_OUTPUT_PATH = ROOT / "accuracy_vs_ec_plots"
PLOT_HISTOGRAM = ROOT / "histogram_plot"
if not OUT.exists():
    os.makedirs(OUT, exist_ok=True)
if not MODEL_HT_PARAMS.exists():
    os.makedirs(MODEL_HT_PARAMS, exist_ok=True)
if not PLOT_OUTPUT_PATH.exists():
    os.makedirs(PLOT_OUTPUT_PATH, exist_ok=True)

# names and types
DATASET_NAMES = ["House", "Bike", "Wine", "Breast-Cancer", "Diabetics", "Cancer"]
MODEL_NAMES = ["Lasso", "Ridge", "RF", "Knn", "SVR"]
ECMethod = Literal["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                    "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]
EC_METHODS: List[ECMethod] = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                    "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]

# analysis constants
SEED = 42
TEST_SIZE = 0.2
