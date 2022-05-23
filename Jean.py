#!/usr/bin/env python3

import numpy as np

narr = np.ndarray


def initPopulation(size: int, P: float = 0.5) -> narr:
    return np.random.rand(size, 8520) < P


def evalPopulation(pop: narr, weights: narr, maxPunish: float) -> narr:
    # get average tfidf score of words in each member
    wordScore = np.mean(pop * weights, axis=1)
    # calculate punishment for ammount of words over minimum
    wordCount = np.sum(pop, axis=1) - 1000
    punishment = wordCount / (8520 - 1000) * maxPunish
    return wordScore - punishment


def chooseSurvivors(pop: narr, scores: narr) -> narr:
    pass


def breedPair(pair: narr) -> narr:
    pass


def breedSurvivors(pop: narr, P: float = 0.6) -> narr:
    pass
