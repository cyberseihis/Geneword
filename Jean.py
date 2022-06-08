#!/usr/bin/env python3

import numpy as np
from collections.abc import Callable
from functools import partial

narr = np.ndarray
ngen = np.random._generator.Generator

rng = np.random.default_rng()


def init_population(size: int, P: float = 0.5) -> narr:
    return np.random.rand(size, 8520) < P


def fitness_score(pop: narr, weights: narr,
                  legal_punish: float, illegal_punish: float) -> narr:
    # get average tfidf score of words in each member
    word_count = np.count_nonzero(pop, axis=1)
    wordScore = np.sum(pop * weights, axis=1) / word_count
    # calculate punishment for ammount of words over minimum
    wordCount = np.sum(pop, axis=1) - 1000
    if(wordCount >= 0):
        punishment = wordCount / (8520 - 1000) * legal_punish
    else:
        punishment = -wordCount * illegal_punish
    return wordScore - punishment


# roulete based on score
def roulete_score(scores: narr) -> narr:
    # remove lowest score from other scores
    rebased = scores - np.min(scores)
    probs = rebased / np.sum(rebased)
    return probs


# roulete based on rank
def roulete_rank(scores: narr) -> narr:
    rank = np.ones(scores.size)
    rank[np.argsort(scores)] = np.arange(scores.size)
    probs = rank / np.sum(rank)
    return probs


def survive(pop: narr,
            evaluator: Callable[[narr], narr],
            score2prob: Callable[[narr], narr]) -> narr:
    scores = evaluator(pop)
    probs = score2prob(scores)
    popN = np.size(scores)
    survivor_idx = rng.choice(popN, popN, p=probs)
    return pop[survivor_idx], scores


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
                    survivor: Callable[[narr], narr],
                    breeder: Callable[[narr], narr],
                    mutator: Callable[[narr], narr],
                    ) -> tuple[narr, narr]:
    survivors, scores = survivor(pop)
    babies = breeder(survivors)
    children = mutator(babies)
    return children, scores


# for given parameters run genetic algorithm
# untill stopping condition is met and return final population
# and scores over the generations
def train(pop_size: int, p_breed: float, p_mut: float,
          evaluator: Callable[[narr], narr],
          ) -> tuple[narr, list[float]]:
    pop = init_population(pop_size)
    history = []
    scores = evaluator(pop)
    best_score = np.mean(scores)
    old_best = best_score
    gen_count = 0
    max_gens = 1000
    survivor = partial(survive, evaluator=evaluator, score2prob=roulete_score)
    breeder = partial(breed_survivors, P=p_breed)
    mutator = partial(mutate, P=p_mut)
    next_gen = partial(next_generation,
                       breeder=breeder, mutator=mutator,
                       survivor=survivor)
    # while (best_score >= old_best or gen_count < max_gens):
    while (gen_count < max_gens):
        gen_count += 1
        old_best = best_score
        pop, scores = next_gen(pop)
        best_score = np.mean(scores)
        history.append(best_score)
        if(gen_count % 10 == 0):
            print(f"{gen_count} $:{best_score}")
    return pop, history


# MAYBE CHANGE TO ONLY RETURNING BEST POPULATION
def batch_train(trainer: Callable[[], tuple[narr, list[float]]]):
    return (trainer() for _ in range(10))


def main():
    config_table = [[20, 0.6, 0], [20, 0.6, 0.01]]
    # Created by save_weights of tf_idf.py
    weights = np.load("tfidf_weights.npz")['a']
    # max punish set to the tf_idf score of a member with
    # all words selected, such that a member
    # with all words is scored 0
    evaluator = partial(fitness_score, weights=weights,
                        legal_punish=np.mean(weights),
                        illegal_punish=np.mean(weights))
    trainer = partial(train, evaluator=evaluator)
    # MISSING TENTRAINS
    my_trainers = (partial(trainer, x, y, z) for x, y, z in config_table)
    return my_trainers
