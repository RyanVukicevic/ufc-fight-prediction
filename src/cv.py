

#file holds any functions related to proving the robustness of models though cross validation
#hence the name, cv.py (cross validation)


import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from src.model import make_logreg_pipeline


def cross_validate_logreg(xy_splits: dict, n_splits=5, random_state=42):
    x_train = xy_splits["x_train"].copy()
    y_train = xy_splits["y_train"].copy()

    model = make_logreg_pipeline()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    results_df = pd.DataFrame({
        "fold": range(1, n_splits + 1),
        "train_accuracy": cv_results["train_accuracy"],
        "val_accuracy": cv_results["test_accuracy"],
        "train_auc": cv_results["train_roc_auc"],
        "val_auc": cv_results["test_roc_auc"]
    })

    summary = {
        "mean_train_accuracy": results_df["train_accuracy"].mean(),
        "mean_val_accuracy": results_df["val_accuracy"].mean(),
        "mean_train_auc": results_df["train_auc"].mean(),
        "mean_val_auc": results_df["val_auc"].mean(),
        "std_val_accuracy": results_df["val_accuracy"].std(),
        "std_val_auc": results_df["val_auc"].std()
    }

    return results_df, summary


def print_cv_summary(results_df, summary, model_name="LOGREG CV"):
    print(f"{model_name}:\n")
    print(results_df)
    print()
    print("mean train accuracy:", summary["mean_train_accuracy"])
    print("mean val   accuracy:", summary["mean_val_accuracy"])
    print("mean train roc_auc:", summary["mean_train_auc"])
    print("mean val   roc_auc:", summary["mean_val_auc"])
    print("std  val   accuracy:", summary["std_val_accuracy"])
    print("std  val   roc_auc:", summary["std_val_auc"])