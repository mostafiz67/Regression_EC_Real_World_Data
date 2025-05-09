from typing import Tuple
import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.datasets import load_boston
from sklearn.model_selection import GroupShuffleSplit, train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder, MinMaxScaler
import scipy.stats as st
import warnings


from src.constants import DATA, TEST_SIZE, SEED

WINE_DIR = DATA / "wine_quality_dataset"
BIKE_DIR = DATA / "bike_sharing_dataset"
MOVIE_DIR = DATA / "ml-1m"
PARKINSONS_DIR = DATA / "parkinsons_tele_dataset"
DIABETES_DIR = DATA / "diabetics_dataset"
BREAST_CANCER_DIR = DATA / "breast_cancer_dataset"
CANCER_DIR = DATA / "cancer_dataset"

WINE_DATA = WINE_DIR / "winequality-red.csv"
BIKE_DATA = BIKE_DIR / "hour.csv"
MOVIE_DATA = MOVIE_DIR / "movies.dat"
MOVIE_RATINGS = MOVIE_DIR / "ratings.dat"
MOVIE_USERS = MOVIE_DIR / "users.dat"
PARKINSONS_DATA = PARKINSONS_DIR / "parkinsons_tele.data"
DIABETICS_DATA = DIABETES_DIR / "diabetics.csv"
BREAST_CANCER_DATA = BREAST_CANCER_DIR / "breast_cancer.csv"
CANCER_DATA = CANCER_DIR / "cancer.csv"

warnings.filterwarnings("ignore")


def boston_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    boston = load_boston()
    df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
    df["Target"] = boston.target

    # print(df.describe(include='all'))
    # print(df.dtypes)

    # Cheking the correlation
    # fig, ax = plt.subplots(figsize = (10,10))
    # g= sns.heatmap(df.corr(),ax=ax, annot= True)
    # ax.set_title('Correlation between variables')
    # plt.show()

    # Checking Outliers by the boxplots
    # colList = list(df.columns.values)
    # df.boxplot(colList)
    # plt.show()

    X = df.drop(columns="Target")
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Doing standarscaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()


def wine_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    df = pd.read_csv(WINE_DATA)

    # Removing outliers
    # for x in ['total sulfur dioxide']:
    #     q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    #     intr_qr = q75 - q25

    #     max = q75 + (1.5 * intr_qr)
    #     min = q25 - (1.5 * intr_qr)

    #     df.loc[df[x] < min, x] = np.nan
    #     df.loc[df[x] > max, x] = np.nan

    # for x in ['free sulfur dioxide']:
    #     q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    #     intr_qr = q75 - q25

    #     max = q75 + (1.5 * intr_qr)
    #     min = q25 - (1.5 * intr_qr)

    #     df.loc[df[x] < min, x] = np.nan
    #     df.loc[df[x] > max, x] = np.nan

    # colList = list(df.columns.values)
    # df.boxplot(colList)
    # plt.show()

    # print(df.isnull().sum()) # No null
    # print(df.describe(include='all'))
    # df = df.fillna(df.mean())
    # print(df.isnull().sum()) # No null

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Doing standarscaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()


def bike_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    # Data preprocessing inherited from https://www.kaggle.com/harshavarshney/bike-sharing-demand-analysis-regression on 25 Sep 2021
    df = pd.read_csv(BIKE_DATA)

    df = df.rename(columns={'weathersit': 'weather',
                            'yr': 'year',
                            'mnth': 'month',
                            'hr': 'hour',
                            'hum': 'humidity',
                            'cnt': 'count'})
    df = df.drop(columns=['instant' , 'dteday' , 'year'])
    df_oh = df

    def one_hot_encoding(data, column):
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
        data = data.drop([column], axis=1)
        return data

    cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']

    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)
    X = X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'], axis=1)
    y = df_oh['count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Doing standarscaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()


def movie_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    movie_df = pd.read_csv(
        MOVIE_DATA, sep="::", names=["MovieID", "Title", "Genres"], engine="python", encoding="latin-1"
    )
    ratings_df = pd.read_csv(
        MOVIE_RATINGS, sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], engine="python", encoding="latin-1"
    )
    user_df = pd.read_csv(
        MOVIE_USERS,
        sep="::",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        engine="python",
        encoding="latin-1",
    )

    df_merged1 = movie_df.merge(ratings_df, how="outer")
    df_merged2 = user_df.merge(ratings_df, how="inner")
    df_merged3 = df_merged1.merge(df_merged2, how="inner")

    df_merged3.UserID = df_merged3.UserID.astype(int)
    df_merged3.Rating = df_merged3.Rating.astype(int)
    df_merged3.sort_values(by=["UserID"], ascending=True)
    master_data = df_merged3[
        ["UserID", "MovieID", "Title", "Rating", "Genres", "Zip-code", "Gender", "Age", "Occupation", "Timestamp"]
    ]
    master_data["Gender"] = master_data["Gender"].apply(lambda v: int(v == "M")).copy()
    master_data.loc[:, "Genres_list"] = master_data["Genres"].str.split("|").copy()
    list2series = pd.Series(master_data.Genres_list)

    mlb = MultiLabelBinarizer()
    one_hot_genres = pd.DataFrame(mlb.fit_transform(list2series), columns=mlb.classes_, index=list2series.index)
    master_features = pd.merge(master_data, one_hot_genres, left_index=True, right_index=True)
    final_movie_df = master_features.drop(columns=["Genres", "Genres_list", "Zip-code"])  # one-hotted already

    final_movie_df = final_movie_df.drop(columns=["Title"])  # Removing title column
    final_movie_df = final_movie_df.drop(columns=["Timestamp"])  # Removing Timestamp column
    X = final_movie_df.drop(columns="Rating")
    y = final_movie_df["Rating"]

    gs = GroupShuffleSplit(n_splits=2, test_size=TEST_SIZE, random_state=SEED)
    train_ix, test_ix = next(gs.split(X, y, groups=X.UserID))

    X_train = X.iloc[train_ix].to_numpy()
    y_train = y.iloc[train_ix].to_numpy().astype(float)
    X_test = X.iloc[test_ix].to_numpy()
    y_test = y.iloc[test_ix].to_numpy().astype(float)

    # Doing standarscaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()

def breast_cancer_dataset():
    # This code is inherited from https://www.kaggle.com/abhash896/machine-learning-models-on-breast-cancer-dataset on Sep 22 2021
    breast_cancer_df = pd.read_csv(BREAST_CANCER_DATA)
    breast_cancer_df = breast_cancer_df.drop('Unnamed: 32', axis=1)  # Drop column
    # Splitting dataset into X and y
    X = breast_cancer_df.drop(columns=["diagnosis", "id"])
    y = breast_cancer_df["diagnosis"]
    y = pd.DataFrame(y)

    # Droping clumns based on the correlation
    corr = X.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                columns[j] = False
    selected_columns = X.columns[columns]
    X = X[selected_columns]

    y.diagnosis = LabelEncoder().fit_transform(y.diagnosis)  # Encoading into 0/1
    y = np.squeeze(y)
    X[["radius_mean", "texture_mean"]] = MinMaxScaler().fit_transform(X[["radius_mean", "texture_mean"]])  # Scalling

    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    return X_train, X_test, y_train.ravel(), y_test.ravel()


def parkinsons_dataset():
    parkinsons_df = pd.read_csv(PARKINSONS_DATA)
    # print(parkinsons_df["motor_UPDRS"].describe())

    # Splitting dataset into X and y
    X = parkinsons_df.drop(columns=["motor_UPDRS"])
    y = parkinsons_df["motor_UPDRS"]
    y = pd.DataFrame(y)

    # Droping clumns based on the correlation
    corr = X.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                columns[j] = False
    selected_columns = X.columns[columns]
    X = X[selected_columns]
    X = X.rename(columns={'subject#': 'subject_id'})

    gs = GroupShuffleSplit(n_splits=2, test_size=TEST_SIZE, random_state=SEED)
    train_ix, test_ix = next(gs.split(X, y, groups=X.subject_id))

    y = np.squeeze(y)
    X_train = X.iloc[train_ix].to_numpy()
    y_train = y.iloc[train_ix].to_numpy().astype(float)
    X_test = X.iloc[test_ix].to_numpy()
    y_test = y.iloc[test_ix].to_numpy().astype(float)

    # Doing standarscaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()

def diabetics_dataset():
    # Data Preprocess code is inherited from https://www.kaggle.com/shrutimechlearn/step-by-step-diabetes-classification-knn-detailed on Sep 23 2021
    diabetics_df = pd.read_csv(DIABETICS_DATA)

    diabetes_data_copy = diabetics_df.copy(deep=True)
    diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[[
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
    diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
    diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
    diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
    diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)
    X = pd.DataFrame(StandardScaler().fit_transform(diabetes_data_copy.drop(["Outcome"], axis=1),),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                              'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = diabetes_data_copy.Outcome

    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Doing standarscaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()

def cancer_dataset():
    # Data Preprocessing and features are inherited from https://www.kaggle.com/dhemanth/cancer-dataset/notebook on Sep 22 2021
    cancer_df = pd.read_csv(CANCER_DATA)

    # Doing some preprocessing
    cancer_df['binnedInc'] = cancer_df['binnedInc'].str.replace('(', '')
    cancer_df['binnedInc'] = cancer_df['binnedInc'].str.replace('[', '')
    cancer_df['binnedInc'] = cancer_df['binnedInc'].str.replace(']', '')
    x = cancer_df['binnedInc'].str.split(',', expand=True).astype(float)
    x = cancer_df['binnedInc'].str.split(',', expand=True).astype(float)
    y = (x[0] + x[1]) / 2
    cancer_df['binnedInc'] = y

    # since target variable has outliers less then 10% of the data, drop the outliers
    df1 = cancer_df[(cancer_df['TARGET_deathRate'] > cancer_df['TARGET_deathRate'].quantile(0.25) - (1.5 * (st.iqr(cancer_df['TARGET_deathRate']))))
                    & (cancer_df['TARGET_deathRate'] < cancer_df['TARGET_deathRate'].quantile(0.75) + (1.5 * (st.iqr(cancer_df['TARGET_deathRate']))))]
    # since 'PctEmployed16_Over' have missing value less then 10% it is imputed with median
    df1['PctEmployed16_Over'] = df1['PctEmployed16_Over'].fillna(df1['PctEmployed16_Over'].median())

    to_t = df1[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over',
                'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate']]
    to_t['avgDeathsPerYear'] = np.log((to_t['avgDeathsPerYear']))
    to_t['avgAnnCount'] = np.log((to_t['avgAnnCount']))
    to_t['popEst2015'] = np.log((to_t['popEst2015']))
    to_t['PctBachDeg25_Over'] = np.log((to_t['PctBachDeg25_Over']))
    to_t['PctOtherRace'] = (np.log((to_t['PctOtherRace']) + 1))
    to_t['BirthRate'] = np.sqrt((to_t['BirthRate']))
    to_t['PercentMarried'] = ((to_t['PercentMarried'])**2)
    to_t['PctMarriedHouseholds'] = ((to_t['PctMarriedHouseholds'])**2)

    y = df1['TARGET_deathRate']
    
    to_t = to_t.values
    X_train, X_test , y_train, y_test = train_test_split(to_t, y, test_size=TEST_SIZE, random_state=SEED)
    return X_train, X_test, y_train.ravel(), y_test.ravel()


if __name__ == "__main__":
    parkinsons_dataset()
