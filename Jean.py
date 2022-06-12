#!/usr/bin/env python3

import numpy as np
from collections.abc import Callable
from functools import partial

narr = np.ndarray
ngen = np.random._generator.Generator

rng = np.random.default_rng()


def random_population(size: int, P: float = 0.5) -> narr:
    return np.random.rand(size, 8520) < P


def first_gen(size: int) -> narr:
    per_member = np.random.rand(size)
    return np.vstack([np.random.rand(8520) < p for p in per_member])


def fitness_score(pop: narr, weights: narr,
                  legal_punish: float, illegal_punish: float) -> narr:
    # get average tfidf score of words in each member
    word_count = np.clip(np.count_nonzero(pop, axis=1), 1, 8520)
    wordScore = np.sum(pop * weights, axis=1) / word_count
    # calculate punishment for ammount of words over minimum
    wordCount = np.sum(pop, axis=1) - 1000
    punishment = np.max([wordCount / (8520 - 1000) * legal_punish,
                        -wordCount * illegal_punish], axis=0)
    return wordScore - punishment


# roulete based on score
def roulete_score(scores: narr) -> narr:
    # remove lowest score from other scores
    rebased = scores - np.min(scores) + (np.max(scores) * 0.01)
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
def breed(pop: narr,
          crossover: Callable[[narr], narr], P: float = 0.6,) -> narr:
    n_pairs = rng.binomial(np.size(pop, axis=0), P) // 2
    parents = pop[0:n_pairs * 2]
    return np.vstack((crossover(parents), pop[n_pairs * 2:]))


def crossover_uniform(pop: narr) -> narr:
    n_pairs = pop.shape[0] // 2
    mask = random_population(n_pairs)
    parents = np.array_split(pop, 2)
    child1 = np.choose(mask, parents)
    child2 = np.choose(~mask, parents)
    return np.vstack((child1, child2))


def crossover_single_pair(pair: narr):
    cut_point = rng.integers(np.shape(pair)[1])
    divided = np.hsplit(pair, [cut_point, pair.size])
    children = np.hstack([divided[0], np.flipud(divided[1])])
    return children


def crossover_single(pop: narr) -> narr:
    n_pairs = pop.shape[0] // 2
    pairs = np.array_split(pop, n_pairs)
    return np.vstack([crossover_single_pair(pair) for pair in pairs])


# Given a population return mutated population
def mutate(pop: narr, P: float = 0.01) -> narr:
    mask = random_population(pop.shape[0], P)
    return pop ^ mask


def elitism(pop: narr, scores: narr, N: int = 2) -> tuple[narr, narr]:
    """Receives a population and the score of each member
    and returns the N best members of the population and the rest
    separated"""
    ind = np.argsort(scores)
    elites = pop[ind[-N:]]
    plebians = pop[ind[:-N]]
    return elites, plebians


# Given a population run the genetic algorithm for one generation
# and return the next population
def next_generation(pop: narr,
                    survivor: Callable[[narr], narr],
                    breeder: Callable[[narr], narr],
                    mutator: Callable[[narr], narr],
                    do_elitism: bool,
                    ) -> tuple[narr, narr]:
    survivors, scores = survivor(pop)
    if(do_elitism):
        elites, survivors = elitism(survivors, scores)
    babies = breeder(survivors)
    children = mutator(babies)
    if(do_elitism):
        children = np.vstack([elites, children])
    # note: under elitism scores are not aligned with the
    # members that achieved them
    return children, scores


# for given parameters run genetic algorithm
# untill stopping condition is met and return final population
# and scores over the generations
def train(pop_size: int, p_breed: float, p_mut: float,
          evaluator: Callable[[narr], narr],
          crossover: Callable[[narr], narr],
          ) -> tuple[narr, list[float]]:
    # pop = first_gen(pop_size)
    pop = first_gen(pop_size)
    history = []
    scores = evaluator(pop)
    best_score = np.mean(scores)
    old_best = best_score
    gen_count = 0
    max_gens = 1000
    stale_count = 0
    survivor = partial(survive,
                       evaluator=evaluator, score2prob=roulete_rank)
    breeder = partial(breed, P=p_breed, crossover=crossover)
    mutator = partial(mutate, P=p_mut)
    next_gen = partial(next_generation,
                       breeder=breeder, mutator=mutator,
                       survivor=survivor, do_elitism=False)
    # while (best_score >= old_best or gen_count < max_gens):
    while (gen_count < max_gens):
        old_best = best_score
        pop, scores = next_gen(pop)
        best_score = np.mean(scores)
        history.append(best_score)
        # if(gen_count % 200 == 0):
        # print(f"{gen_count} $:{best_score:.7f} #:{np.sum(pop[0])}")
        gen_count += 1
        stale_count = stale_count + 1 if old_best >= best_score else 0
        if stale_count > 20:
            # print(f"{gen_count} $:{best_score:.7f} #:{np.sum(pop[0])}")
            break
    champion, _ = elitism(pop, scores, N=1)
    return champion, history


# MAYBE CHANGE TO ONLY RETURNING BEST POPULATION
def batch_train(trainer: Callable[[], tuple[narr, list[float]]],
                ):
    batch_results = [trainer() for _ in range(10)]
    return batch_results


def merge_results(results: list[tuple[narr, list]],
                  evaluator: Callable[[narr], narr]
                  ) -> tuple[narr, list]:
    gladiators = [a[0] for a in results]
    meta_history = [a[1] for a in results]
    glad_pop = np.vstack(gladiators)
    glad_scores = evaluator(glad_pop)
    champion, _ = elitism(glad_pop, glad_scores, N=1)
    return champion, meta_history


def main():
    config_table = [
                    (20, 0.6, 0.00),
                    (20, 0.6, 0.01),
                    (20, 0.6, 0.10),
                    (20, 0.9, 0.01),
                    (20, 0.1, 0.01),
                    (200, 0.6, 0.00),
                    (200, 0.6, 0.01),
                    (200, 0.6, 0.10),
                    (200, 0.9, 0.01),
                    (200, 0.1, 0.01),
                    ]
    # Created by save_weights of tf_idf.py
    weights = np.load("tfidf_weights.npz")['a']
    # max punish set to the tf_idf score of a member with
    # all words selected, such that a member
    # with all words is scored 0
    evaluator = partial(fitness_score, weights=weights,
                        legal_punish=np.mean(weights),
                        illegal_punish=np.mean(weights))
    trainer = partial(train, evaluator=evaluator,
                      crossover=crossover_uniform)
    # MISSING TENTRAINS
    my_trainers = (partial(trainer, x, y, z) for x, y, z in config_table)
    batches = (batch_train(tr) for tr in my_trainers)
    merger = partial(merge_results, evaluator=evaluator)
    per_batch = [merger(b) for b in batches]
    champion, complete_history = merger(per_batch)
    complete_dict = dict(zip(config_table, complete_history))
    return champion, complete_dict
