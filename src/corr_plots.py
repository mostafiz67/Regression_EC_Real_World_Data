"""
Some code is inherited from https://stackoverflow.com/questions/68123724/how-to-plot-multiple-csv-files-with-separate-plots-for-each-category
"""

from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from typing import List
import math

# from src.constants import OUT, PLOT_OUTPUT_PATH

# CSVS = [
#     OUT / "House_error.csv",
#     OUT / "Bike_error.csv",
#     OUT / "Wine_error.csv",
# ]

ECMethods = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed"]

def correlation_analysis() -> DataFrame:
        corrs = []
        # csv = pd.read_csv()
        # error_kind = csv.stem[: csv.stem.find("_")]
        main_df = pd.read_csv("/home/mostafiz/Desktop/Regression_EC/output/Bike_error.csv").drop(columns=["Unnamed: 0", "k", "n_rep", "EC_vec_sd", "EC_scalar_sd", "MAE_sd", "MAPE", "MAPE_sd", "MSqE_sd", "R2_sd"])
        for i in range(len(ECMethods)):
            df = main_df.loc[main_df['Method'] == ECMethods[i]]
            print(df)
            # Method one using FacetGrid
            g = sns.FacetGrid(df, col="Regressor", col_wrap=3)
            g.map_dataframe(lambda data, color:sns.heatmap(df.corr(), annot=True, fmt='.2f', square=True))
            # fig_name = "figure_'{0}'_method_'{1}'.png".format(error_kind, ECMethods[i])
            # outfile = os.path.join(PLOT_OUTPUT_PATH, 'corr_plot/') + fig_name
            plt.show()
            # plt.savefig(outfile, format='png')
            plt.close()
                
        corr_df = pd.concat(corrs, axis=1)
        return corr_df



if __name__ == "__main__":
    corr_df = correlation_analysis()
    # corr_outfile = OUT / "Datasets_error_correlations.csv"
    # corr_df.to_csv(corr_outfile)
    print(corr_df)
    


























# CSVS = [
#     OUT / "House_error.csv",
#     OUT / "Bike_error.csv",
#     OUT / "Wine_error.csv",
# ]

# ECMethods = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed"]

# def correlation_analysis() -> DataFrame:
#         corrs = []
#         for csv in CSVS:
#             error_kind = csv.stem[: csv.stem.find("_")]
#             main_df = pd.read_csv(csv).drop(columns=["Unnamed: 0"])
#             for i in range(len(ECMethods)):
#                 df = main_df.loc[main_df['Method'] == ECMethods[i]]

#                 # Method one using FacetGrid
#                 g = sns.FacetGrid(df, col="Regressor", col_wrap=2)
#                 g.map_dataframe(lambda data, color:sns.heatmap(data.corr(), annot=True, fmt='.2f', square=True))
#                 fig_name = "figure_'{0}'_method_'{1}'.png".format(error_kind, ECMethods[i])
#                 outfile = os.path.join(PLOT_OUTPUT_PATH, 'corr_plot/') + fig_name
#                 plt.show()
#                 plt.savefig(outfile, format='png')
#                 plt.close()
                
#         corr_df = pd.concat(corrs, axis=1)
#         return corr_df



# if __name__ == "__main__":
#     corr_df = correlation_analysis()
#     corr_outfile = OUT / "Datasets_error_correlations.csv"
#     # corr_df.to_csv(corr_outfile)
#     print(corr_df)
    