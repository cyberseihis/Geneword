import pickle
import numpy as np
import pandas as pd


def score(hist):
    return hist[-1]


def duration(hist):
    return len(hist)


def meta2res(meta):
    avg_score = np.mean([score(i) for i in meta])
    avg_duration = np.mean([duration(i) for i in meta])
    return avg_score, avg_duration


def get_table():
    with open("genetics.pkl", 'rb') as fi:
        _, complete_history = pickle.load(fi)
    meta_table = [[*i, *meta2res(met)]
                  for i, met in complete_history.items()]
    df = pd.DataFrame(
        meta_table,
        columns=["Population Size",
                 "Crossover Propability",
                 "Mutation Propability",
                 "Average Score",
                 "Average Duration"])
    print(df.to_markdown())


get_table()
