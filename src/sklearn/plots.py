import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix


def plot_countplot(df, column, user_friendly_column_name, rotation=0):
    print("\n-----------------------------------------------------")
    print(f"\n{user_friendly_column_name} Distribution")
    palette = "deep"
    sns.set_palette(palette)

    sns.countplot(data=df, x=column)

    plt.xlabel(f"{user_friendly_column_name}")
    plt.ylabel("Number of Records")
    plt.title(f"{user_friendly_column_name} Distribution")
    plt.xticks(rotation=rotation)

    plt.show()


def plot_displot(df, column, user_friendly_column_name, rotation=0, bins=20):
    print("\n-----------------------------------------------------")
    print(f"\n{user_friendly_column_name} Distribution")
    palette = "deep"
    sns.set_palette(palette)

    sns.displot(data=df, x=column, kde=True, bins=bins)

    plt.xlabel(f"{user_friendly_column_name}")
    plt.ylabel("Number of Records")
    plt.title(f"{user_friendly_column_name} Distribution")
    plt.xticks(rotation=rotation)

    plt.show()


def plot_stacked_bar(df, column1, column2, rotation=0):
    print("\n-----------------------------------------------------")
    print(f"\n{column1} & {column2} Distribution")
    palette = "deep"
    sns.set_palette(palette)

    pd.crosstab(df[column1], df[column2]).plot(kind="bar", stacked=True)

    plt.xlabel(f"{column1}")
    plt.ylabel("Number of Records")
    plt.title(f"{column1} & {column2} Distribution")
    plt.xticks(rotation=rotation)

    plt.show()


def evaluate_model(y_test, y_pred):
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print("\n---------------------------------------------\n")
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, cmap="Greens", fmt=".0f")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
