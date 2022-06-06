#!/usr/bin/env python3

import numpy as np

narr = np.ndarray
ngen = np.random._generator.Generator

rng = np.random.default_rng()


def init_population(size: int, P: float = 0.5) -> narr:
    return np.random.rand(size, 8520) < P


def fitness_score(pop: narr, weights: narr, maxPunish: float) -> narr:
    # get average tfidf score of words in each member
    wordScore = np.mean(pop * weights, axis=1)
    # calculate punishment for ammount of words over minimum
    wordCount = np.sum(pop, axis=1) - 1000
    punishment = wordCount / (8520 - 1000) * maxPunish
    return wordScore - punishment


# roulete based on score
def survivor_index(scores: narr) -> narr:
    probs = scores / np.sum(scores)
    popN = np.size(scores)
    survivors = rng.choice(popN, popN, p=probs)
    return survivors


def survivor_population(pop: narr, idx: narr) -> narr:
    return pop[idx]


def breed_survivors(pop: narr, P: float = 0.6) -> narr:
    pair_num = rng.binomial(np.size(pop, axis=0), P) // 2
    parents1 = pop[0:pair_num]
    parents2 = pop[pair_num:pair_num * 2]
    mask = init_population(pair_num)
    child1 = np.choose(mask, (parents1, parents2))
    child2 = np.choose(mask, (parents2, parents1))
    return np.vstack((child1, child2, pop[pair_num * 2:]))


def mutate(pop: narr, P: float = 0.01) -> narr:
    mask = init_population(pop.shape[0], P)
    return pop ^ mask
