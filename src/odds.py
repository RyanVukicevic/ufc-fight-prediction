


#file holds functions relating to odds ratios, quantifiers of feature importance especially in 
#logistic regression contexts 

import pandas as pd

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


def get_feature_ranking_by_coef(odds_df: pd.DataFrame):

    ranking_by_coef = odds_df.reindex(
    odds_df["coefficient"].abs().sort_values(ascending=False).index)

    odds_df["coefficient"].round(4)
    odds_df["odds_ratio"].round(4)
    
    return ranking_by_coef
    