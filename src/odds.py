
def get_logreg_odds_ratios(model, feature_names):
    import numpy as np
    import pandas as pd

    # extract the trained logistic regression inside pipeline
    clf = model.named_steps["clf"]

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": odds_ratios
    })

    return df.sort_values(by="odds_ratio", ascending=False)