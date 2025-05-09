"""
Author: Md Mostafizur Rahman
File: Calculate Regression Error Consistency using different methods and benchmark datasets
"""

from argparse import Namespace
from itertools import combinations, count, repeat
from typing import Any, Dict, List, Tuple
from warnings import filterwarnings
# import matplotlib.pyplot as plt
# import seaborn as sns

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm.contrib.concurrent import process_map

from src.constants import ECMethod, PLOT_HISTOGRAM


def regression_ec(residuals: List[ndarray], method: ECMethod) -> List[ndarray]:
    filterwarnings("ignore", "invalid value encountered in true_divide", category=RuntimeWarning)
    consistencies = []
    for pair in combinations(residuals, 2):
        r1, r2 = pair
        r = np.vstack(pair)
        sign = np.sign(np.array(r1) * np.array(r2))
        if method == "ratio-signed":
            consistency = np.multiply(sign, np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0))
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio":
            consistency = np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0)
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio-diff-signed":
            consistency = np.multiply(sign, (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2)))
            consistency[np.isnan(consistency)] = 0
        elif method == "ratio-diff":
            consistency = (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2))
            consistency[np.isnan(consistency)] = 0
        elif method =="intersection_union_sample":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choice_numerator = [np.minimum(r1, r2), np.zeros(len(r1))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator)
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choice_numerator = [np.minimum(r1, r2), np.zeros(len(r1))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator)
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="negative_incon":
            conditions = [(r1>=0)&(r2>=0), (r1<0)&(r2<0)]
            choice_numerator = [np.minimum(r1, r2), -np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator, -np.minimum(np.abs(r1), np.abs(r2)))
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="positive_incon":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choiceValue = [np.minimum(np.abs(r1), np.abs(r2)), np.add(np.abs(r1), np.abs(r2))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        elif method =="intersection_union_distance":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.add(np.abs(r1), np.abs(r2))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    filterwarnings("default", "invalid value encountered in true_divide", category=RuntimeWarning)
    return consistencies


def ec_inner_loop(args: Namespace) -> Dict[str, DataFrame]:
    dataset = args.dataset
    regressor = args.regressor
    reg_name = args.reg_name
    k = args.k
    X_train, X_test, y_train, y_test = dataset

    fold_residuals, fold_dfs = [], []
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, _ in kf.split(X_train):
        if reg_name in ["Lasso"]:
            preds = regressor.fit(X_train[train_index], y_train[train_index].reshape(-1, 1)).predict(X_test)
        else:
            preds = regressor.fit(X_train[train_index], y_train[train_index]).predict(X_test)
        resid = preds - y_test
        fold_residuals.append(resid)
        fold_df = pd.DataFrame()
        fold_df["MSqE"] = [mean_squared_error(y_test, preds)]
        fold_df["RMSqE"] = [mean_squared_error(y_test, preds, squared=False)]
        fold_df["MAE"] = [mean_absolute_error(y_test, preds)]
        fold_df["MAPE"] = [np.mean(np.abs(preds - y_test) / (y_test + 1e-5))]
        fold_df["R2"] = [r2_score(y_test, preds)]
        fold_dfs.append(fold_df)
    rep_df = pd.concat(fold_dfs, axis=0, ignore_index=True)
    return fold_residuals, rep_df

# def plot_histogram_residuals(rep_residual, reg_name):
#     # print(np.shape(rep_residual.mean(axis=0)))
#     plt.hist(rep_residual.mean(axis=0).T, orientation='horizontal', bins=10)
#     plt.xlabel("Samples")
#     plt.ylabel(f"Residuals Value of the {reg_name} model")
#     plt.title(f"All residuals vs Samples (parkinson-tele Dataset)")
#     plt.savefig(PLOT_HISTOGRAM / "residual_plot/parkinson-tele" / f"residual_hist_parkinson-tele_{reg_name}.png", bbox_inches='tight')
#     plt.close()

# def plot_histogram_ec(consistencies, method, reg_name):
#     # print(np.shape(consistencies.mean(axis=0)))
#     plt.hist(consistencies.mean(axis=0).T, orientation='horizontal', bins=10)
#     plt.xlabel("Samples")
#     plt.ylabel(f"EC Value method ({method})")
#     plt.title(f"Histogram of Error Consistency ({method}) for {reg_name} vs Samples (parkinson-tele Dataset)")
#     plt.savefig(PLOT_HISTOGRAM / "ec_plot/parkinson-tele" / f"ec_hist_parkinson-tele_{method}_{reg_name}.png", bbox_inches='tight')
#     plt.close()

def calculate_ECs(
    dataset: Tuple[ndarray, ndarray, ndarray, ndarray],
    regressor: Any,
    reg_name: Any,
    methods: List[ECMethod],
    k: int,
    repetitions: int,
    pbar: bool = False,
) -> DataFrame:
    rep_residuals, rep_stats = [], []
    args = Namespace(
        **dict(dataset=dataset, regressor=regressor, reg_name=reg_name, methods=methods, k=k, repetitions=repetitions)
    )
    rep_residuals, rep_stats = list(
        zip(
            *process_map(
                ec_inner_loop,
                repeat(args, repetitions),
                total=repetitions,
                desc=f"Computing ECs for {reg_name:.<10}",
                disable=not pbar,
                max_workers=8,
            )
        )
    )
    stats = pd.concat(list(rep_stats), axis=0)
    print(stats)
    summaries = []
    
    rep_residuals = np.array(rep_residuals).reshape(k*repetitions, -1) # shape(rep_residual)== (nb_rep, nb_fold, test_samples)
    # print(np.shape(rep_residuals)) # (nb_rep*nb_fold, nb_samples)
    # plot_histogram_residuals(rep_residuals, reg_name)
    for method in methods:
        consistencies: ndarray = np.array(regression_ec(list(rep_residuals), method))
        # plot_histogram_ec(consistencies, method, reg_name)
        # print(consistencies)
        # print(method, consistencies.shape, (consistencies.mean()))
        stats["EC"]= consistencies.mean()
        stats["Method"] = method
        print(stats) 
        summaries.append(
            pd.DataFrame(
                {
                    "Regressor": reg_name,
                    "Method": method,
                    "k": k,
                    "n_rep": repetitions,
                    "EC": consistencies.mean(),
                    "EC_vec_sd":  consistencies.std(ddof=1) if method == "intersection_union_all" else consistencies.mean(axis=0).std(ddof=1),
                    "EC_scalar_sd": "NA" if method == "intersection_union_all" else consistencies.mean(axis=1).std(ddof=1),
                    "EC_median": np.median(consistencies),
                    "EC_1%": np.percentile(consistencies, 1),
                    "EC_5%": np.percentile(consistencies, 5),
                    "EC_25%": np.percentile(consistencies, 25),
                    "EC_50%": np.percentile(consistencies, 50),
                    "EC_75%": np.percentile(consistencies, 75),
                    "EC_95%": np.percentile(consistencies, 95),
                    "EC_99%": np.percentile(consistencies, 99),
                    "EC_max": np.max(consistencies),
                    "EC_min": np.min(consistencies),
                    "MAE": np.mean(stats["MAE"]),
                    "MAE_sd": np.std(stats["MAE"], ddof=1),
                    "MAPE": np.mean(stats["MAPE"]),
                    "MAPE_sd": np.std(stats["MAPE"], ddof=1),
                    "MSqE": np.mean(stats["MSqE"]),
                    "MSqE_sd": np.std(stats["MSqE"], ddof=1),
                    "RMSqE": np.mean(stats["RMSqE"]),
                    "RMSqE_sd": np.std(stats["RMSqE"], ddof=1),
                    "R2": np.mean(stats["R2"]),
                    "R2_sd": np.std(stats["R2"], ddof=1),
                },
                index=[0],
            )
        )
    summary = pd.concat(summaries, axis=0, ignore_index=True)
    return summary