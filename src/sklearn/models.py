from xgboost import XGBClassifier

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def baseline_classifier(x_train, y_train):
    dummy = DummyClassifier(strategy="uniform", random_state=42)
    dummy.fit(x_train, y_train)
    print(f"Dummy Classifier Accuracy: {dummy.score(x_train, y_train)}")


def evaluate_classifiers(x_train, y_train, classifiers):
    """
    Evaluate the performance of different classifiers
    ex)
        classifiers = [
            ("Decision Tree", DecisionTreeClassifier()),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", XGBClassifier(tree_method="hist")),
            ("ada", AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)),
        ]
    """
    # Iterate over each classifier and evaluate performance
    for clf_name, clf in classifiers:
        scores = cross_val_score(clf, x_train, y_train, cv=5, scoring="accuracy")

        # Calculate average performance metrics
        avg_accuracy = scores.mean()
        avg_precision = cross_val_score(
            clf, x_train, y_train, cv=5, scoring="precision_macro"
        ).mean()
        avg_recall = cross_val_score(
            clf, x_train, y_train, cv=5, scoring="recall_macro"
        ).mean()

        # Print the performance metrics
        print(f"Classifier: {clf_name}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print("-----------------------")


def format_classifier_metrics(clf_name, avg_accuracy, avg_precision, avg_recall):
    print(f"Classifier: {clf_name}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print("-----------------------")
