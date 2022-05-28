#!/usr/bin/env python3

import numpy as np

narr = np.ndarray
ngen = np.random._generator.Generator


def initPopulation(size: int, P: float = 0.5) -> narr:
    return np.random.rand(size, 8520) < P


def fitnessScore(pop: narr, weights: narr, maxPunish: float) -> narr:
    # get average tfidf score of words in each member
    wordScore = np.mean(pop * weights, axis=1)
    # calculate punishment for ammount of words over minimum
    wordCount = np.sum(pop, axis=1) - 1000
    punishment = wordCount / (8520 - 1000) * maxPunish
    return wordScore - punishment


# roulete based on score
def chooseSurvivors(scores: narr, rng) -> narr:
    probs = scores / np.sum(scores)
    rng = np.random.default_rng()
    popN = np.size(scores)
    survivors = rng.choice(popN, popN, p=probs)
    return survivors


def breedPair(pair: narr) -> narr:
    pass


def breedSurvivors(pop: narr, P: float = 0.6) -> narr:
    pass
