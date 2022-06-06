#!/usr/bin/env python3

import numpy as np
from collections.abc import Callable
from functools import partial

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


# Given a population select some of them for breeding and return
# a population with the parents replaced by the children
def breed_survivors(pop: narr, P: float = 0.6) -> narr:
    pair_num = rng.binomial(np.size(pop, axis=0), P) // 2
    parents1 = pop[0:pair_num]
    parents2 = pop[pair_num:pair_num * 2]
    mask = init_population(pair_num)
    child1 = np.choose(mask, (parents1, parents2))
    child2 = np.choose(mask, (parents2, parents1))
    return np.vstack((child1, child2, pop[pair_num * 2:]))


# Given a population return mutated population
def mutate(pop: narr, P: float = 0.01) -> narr:
    mask = init_population(pop.shape[0], P)
    return pop ^ mask


# Given a population run the genetic algorithm for one generation
# and return the next population
def next_generation(pop: narr,
                    evaluator: Callable[[narr], narr],
                    breeder: Callable[[narr], narr],
                    mutator: Callable[[narr], narr],
                    ) -> tuple[narr, narr]:
    scores = evaluator(pop)
    survivors = survivor_population(pop, survivor_index(scores))
    hatchlings = breeder(survivors)
    children = mutator(hatchlings)
    return children, scores


# Check if stopping conditions are met
def stopping(scores: narr) -> bool:
    return False


# for given parameters run genetic algorithm
# untill stopping condition is met and return final population
# and scores over the generations
def train(pop_size: int, p_breed: float, p_mut: float,
          evaluator: Callable[[narr], narr],
          ) -> tuple[narr, list[float]]:
    pop = init_population(pop_size)
    history = []
    scores = evaluator(pop)
    breeder = partial(breed_survivors, P=p_breed)
    mutator = partial(mutate, P=p_mut)
    next_gen = partial(next_generation,
                       breeder=breeder, mutator=mutator,
                       evaluator=evaluator)
    while (not stopping(scores)):
        pop, scores = next_gen(pop)
        history.append(scores)
    return pop, history


def main():
    # Created by save_weights of tf_idf.py
    weights = np.load("tfidf_weights.npz")['a']
    # max punish set to the maximul score average of tf_idfs
    # can give to 1000 words
    evaluator = partial(fitness_score, weights=weights,
                        maxPunish=0.002667750770568537)
