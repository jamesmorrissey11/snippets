import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay


def get_column_details(data, column):
    print("Details of", column, "column")
    print("\nDataType: ", data[column].dtype)
    count_null = data[column].isnull().sum()
    if count_null == 0:
        print("\nThere are no null values")
    elif count_null > 0:
        print("\nThere are ", count_null, " null values")
    print("\nNumber of Unique Values: ", data[column].nunique())


def find_missing_columns(dataframe):
    missing_columns = dataframe.columns[dataframe.isnull().any()].tolist()
    return missing_columns


def is_categorical(column, threshold: int = 1000):
    if column.dtype in ["object", "category"]:
        return True
    else:
        num_unique_values = len(column.unique())
        if len(num_unique_values) < threshold:
            return True
        else:
            return False


def find_categorical_columns(dataframe):
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return categorical_columns


def check_for_nulls(data):
    print(data.isnull().sum() / len(data) * 100)


def feature_target_correlation(data: pd.DataFrame, target_col: str):
    print(data.corr()[target_col])
    plt.figure(figsize=(30, 30))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


def plot_highest_correlated_features(data: pd.DataFrame, target_col: str):
    correlations = data.corr()[target_col]
    top_10_features = correlations.abs().nlargest(10).index
    top_10_corr_values = correlations[top_10_features]
    plt.figure(figsize=(10, 11))
    plt.bar(top_10_features, top_10_corr_values)
    plt.xlabel("Features")
    plt.ylabel("Correlation with Target")
    plt.title("Top 10 Features with Highest Correlation to Target")
    plt.xticks(rotation=45)
    plt.show()


def get_highest_correlated_features(data: pd.DataFrame, target_col: str):
    correlations = data.corr()[target_col]
    top_10_features = correlations.abs().nlargest(10).index
    top_10_corr_values = correlations[top_10_features]
    return top_10_features, top_10_corr_values


def plot_confustion_matrix(y_true, preds, filename=False):
    _, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, preds, ax=ax)
    if filename:
        plt.savefig(filename)


def plot_target_variable(df, target_col):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    ax = sns.countplot(df[target_col], palette="Set2")
    ax.bar_label(ax.containers[0], fontweight="black", size=15)
    plt.title(f"{target_col} Disribution", fontweight="black", size=20, pad=20)


def indicies_of_outliers(x: int) -> np.array(int):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))


def baseline_classifier(x_train, y_train):
    dummy = DummyClassifier(strategy="uniform", random_state=42)
    dummy.fit(x_train, y_train)
    print(f"Dummy Classifier Accuracy: {dummy.score(x_train, y_train)}")


def baseline_classifier(x_train, y_train):
    dummy = DummyClassifier(strategy="uniform", random_state=42)
    dummy.fit(x_train, y_train)
    print(f"Dummy Classifier Accuracy: {dummy.score(x_train, y_train)}")
