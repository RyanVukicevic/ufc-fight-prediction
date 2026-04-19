import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


def evaluate_binary_classifier(model, x_train, y_train, x_test, y_test):
    p_train = model.predict_proba(x_train)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    yhat_train = (p_train >= 0.5).astype(int)
    yhat_test = (p_test >= 0.5).astype(int)

    metrics = {
        "train_accuracy": accuracy_score(y_train, yhat_train),
        "test_accuracy": accuracy_score(y_test, yhat_test),
        "train_auc": roc_auc_score(y_train, p_train),
        "test_auc": roc_auc_score(y_test, p_test),
        "fraction_close_calls": ((p_test > 0.45) & (p_test < 0.55)).mean(),
    }

    return metrics


def print_metrics(model_name_str, model, metrics):
    
    print(f"{model_name_str}:\n")

    print(model)

    print("\ntrain accuracy:", metrics["train_accuracy"])
    print("test  accuracy:", metrics["test_accuracy"])
    print("train roc_auc:", metrics["train_auc"])
    print("test  roc_auc:", metrics["test_auc"])
    print("fraction close calls:", metrics["fraction_close_calls"])


def train_logreg(xy_splits: dict):
    x_train = xy_splits["x_train"].copy()
    x_test = xy_splits["x_test"].copy()
    y_train = xy_splits["y_train"].copy()
    y_test = xy_splits["y_test"].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, n_jobs=None))
    ])

    model.fit(x_train, y_train)

    metrics = evaluate_binary_classifier(model, x_train, y_train, x_test, y_test)
    # print_metrics("logreg", metrics)

    return model, metrics


def train_boosted(xy_splits: dict):
    x_train = xy_splits["x_train"].copy()
    x_test = xy_splits["x_test"].copy()
    y_train = xy_splits["y_train"].copy()
    y_test = xy_splits["y_test"].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.03,
            max_iter=1200,
            min_samples_leaf=50,
            l2_regularization=5.0,
            random_state=42
        ))
    ])

    model.fit(x_train, y_train)

    metrics = evaluate_binary_classifier(model, x_train, y_train, x_test, y_test)
    # print_metrics("boosted", metrics)

    return model, metrics