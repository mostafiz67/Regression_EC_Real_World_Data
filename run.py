import argparse
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src import data_preprocess
from src.constants import DATASET_NAMES, EC_METHODS, OUT, MODEL_HT_PARAMS, MODEL_NAMES
from src.regression_EC import calculate_ECs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameter for passing dataset information")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True, help="Enter the name of the dataset")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.dataset == "House":
        load_data = data_preprocess.boston_dataset
    elif args.dataset == "Bike":
        load_data = data_preprocess.bike_dataset
    elif args.dataset == "Wine":
        load_data = data_preprocess.wine_dataset
    elif args.dataset == "MovieLense":
        load_data = data_preprocess.movie_dataset
    elif args.dataset == "Breast-Cancer":
        load_data = data_preprocess.breast_cancer_dataset
    elif args.dataset == "Parkinsons-Tele":
        load_data = data_preprocess.parkinsons_dataset
    elif args.dataset == "Diabetics":
        load_data = data_preprocess.diabetics_dataset
    elif args.dataset == "Cancer":
        load_data = data_preprocess.cancer_dataset
    else:
        raise RuntimeError("Unreachable!")
    dataset = load_data()

    models_params = {}
    for model_name in MODEL_NAMES:
        filename = f"{args.dataset}_{model_name}_params.json"
        outfile = MODEL_HT_PARAMS / filename
        with open(outfile, "r") as file:
            d_loaded = json.load(file)
            models_params[model_name] = d_loaded

    svr_norm = models_params["SVR"].pop("norm")

    REGRESSORS = {
        "LinReg": LinearRegression(),
        "Lasso-H": Lasso(**(models_params.get("Lasso"))),
        "Lasso": Lasso(precompute=True),
        "Ridge-H": Ridge(**(models_params.get("Ridge"))),
        "Ridge": Ridge(solver="lsqr"),
        "RF-H": RandomForestRegressor(**(models_params.get("RF"))),
        "RF": RandomForestRegressor(),
        "Knn-H": KNeighborsRegressor(**(models_params.get("Knn"))),
        "Knn": KNeighborsRegressor(),
        "SVR-H": SVR(**models_params.get("SVR")),
        "SVR": SVR(),
    }

    K = 5
    N_REPS = 50

    dfs = []
    for reg_name, regressor in REGRESSORS.items():
        dfs.append(
            calculate_ECs(
                dataset=dataset,
                regressor=regressor,
                reg_name=reg_name,
                methods=EC_METHODS,
                k=K,
                repetitions=N_REPS,
                pbar=args.progress,
            )
        )
    df = pd.concat(dfs, axis=0, ignore_index=True)
    filename = f"{args.dataset}_error.csv"
    outfile = OUT / filename
    df.to_csv(outfile)
    print(f"Saved results for {args.dataset} error to {outfile}")
