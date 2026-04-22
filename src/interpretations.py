

#this file exists to store functions where i print my interpretations of information found in the pipeline
#serves as a place to store my thoughts after important parts, for future reference, and to display the
#the reasoning to anyone who may see this code 


#after cleaning data and storing in dictionary named cleaned
def interpret_cleaning():

    print("=== FEATURE ENGINEERING INTERPRETATION ===\n")

    print("GENERAL:")
    print("each row represents a fight from fighter A's perspective")
    print("delta_* features are computed as fighter_A_stat - fighter_B_stat")
    print("positive delta means fighter A has more of that stat\n")

    print("EXPERIENCE FEATURES:")
    print("fights_entering: total number of fights a fighter has had before this fight")
    print("wins_entering: total wins before the fight")
    print("losses_entering: total losses before the fight")
    print("these capture overall experience and track record\n")

    print("PERFORMANCE RATES:")
    print("win_rate_entering: wins / total fights, measures overall success")
    print("finish_win_rate_entering: proportion of wins that were finishes")
    print("ko_win_rate_entering: proportion of wins by KO/TKO")
    print("sub_win_rate_entering: proportion of wins by submission")
    print("these capture how effective a fighter is, not just how experienced\n")

    print("FIGHT TYPE / LEVEL FEATURES:")
    print("five_round_fights_entering: number of 5-round fights previously")
    print("five_round_rate_entering: proportion of fights that were 5 rounds")
    print("5-round fights are usually main events or title fights")
    print("so this acts as a proxy for high-level / elite experience\n")

    print("TIME FEATURES:")
    print("avg_fight_time_entering: average fight duration before this fight")
    print("time_elapsed_seconds: duration of the current fight")
    print("pct_of_fight_completed: how much of the scheduled fight was completed")
    print("these capture pacing, endurance, and finishing tendencies\n")

    print("RECENCY FEATURES:")
    print("days_since_last_fight: time since fighter's last fight")
    print("captures activity level, ring rust, and recovery\n")

    print("PHYSICAL ATTRIBUTES:")
    print("height, reach: physical measurements of fighters")
    print("age_at_fight: fighter's age at time of fight")
    print("age is especially important due to physical decline over time\n")

    print("DELTA FEATURES (MODEL INPUT):")
    print("delta_* features compare fighter A vs fighter B")
    print("example: delta_wins_entering = A_wins - B_wins")
    print("these are what the model actually uses to predict outcomes\n")

    print("SUMMARY:")
    print("the model is given differences in experience, performance, physical traits, and fight history")
    print("and learns how those differences impact the probability of winning\n")

#after calling cv and training models
def interpret_cv_and_scores():
    print("\nFrom 4 and 4.5 it can be seen that cv scores in 5-fold approximately\n" \
    "equal to train/test from logreg model\n" \
    "proving its robustness to changes in how the train/test is split")

    print("\nAccuracy around the 60% mark is normal for UFC prediction, many fighters\n" \
    "do not have many fights as the sport is only 30 years old and any given\n" \
    "fighter only fights twice a year or so, so 30% are coinflips for the model\n" \
    "as its confidence lies within [45, 55]% confidence for fights marked as close calls")

    print("\nThe boosted model achieves higher training performance but does not improve\n" \
    "test performance, suggesting mild overfitting\n" \
    "logistic regression performs similarly or better on unseen data while remaining interpretable, so i prefer it")

#after calling fns from odds to see the most influential features
def interpret_odds():
    print("after controlling for Elo, since its coefficient is highest abs val by a\n"
    "decent margin, win rate, age, experience, and five-round exposure,\n" \
    "raw loss/win counts are acting as schedule/experience correction variables")

    print("final interpretation from the top 5 highest weighted features: 'the better, younger, and more experienced (especially in 5-round fights) fighter tends to win'")



    
