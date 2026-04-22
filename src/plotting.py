from matplotlib import pyplot as plt
import pandas as pd

def plot_fights_per_fighter(fighter_stats: pd.DataFrame):

    fighter_counts = fighter_stats["fighter_url"].value_counts()

    fight_count_dist = fighter_counts.value_counts().sort_index()

    plt.figure()

    plt.bar(fight_count_dist.index, fight_count_dist.values, color="darkred")

    plt.xticks(range(0, max(fight_count_dist.index)+1, 2)) 

    plt.xlabel("Number of Fights")
    plt.ylabel("Number of Fighters")
    plt.title("Distribution of Fights per Fighter")

    plt.show()