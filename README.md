# Regression EC with Real Life Machine Learning Dataset 

Used six different Medical and Non-Medical datasets to analyze the newly proposed regression error consistency metrics (EC) metrics with 6 ML regressor (Linear, Lasso, Ridge, RF, Knn, and SVR). All Datasets has been gone through different data preprocessing, and feature engineering steps, while all ML were tuned for every dataset. Here, we used 5-fold external holdout cross-validation and we repeated the process 50 times.

<img src="https://github.com/mostafiz67/Regression-EC-RL-Data/blob/master/accuracy_vs_ec_plots/new_medical_plots/MAE_House_only.png" alt="MAE VS Regression EC (House Dataset)" width="800" height="500">
<p align="center">
    Figure: MAE VS Regression EC (House Dataset)
</p>

<img src="https://github.com/mostafiz67/Regression-EC-RL-Data/blob/master/accuracy_vs_ec_plots/new_medical_plots/MAE_Diabetics_only.png" alt="MAE VS Regression EC (Diabetics Dataset)" width="800" height="500">
<p align="center">
    Figure: MAE VS Regression EC (Diabetics Dataset)
</p>

### Execute
Use `python3 -m run --dataset=Dataset_Name --progress` to execute the main function!

##### Argument Passing
You need to choose the `Dataset_Name` like "Boston", "Bike", etc to execute the program for the respective dataset. Additionally, if you want to see the progress bar then use `--progress`, otherwise ignore it.
