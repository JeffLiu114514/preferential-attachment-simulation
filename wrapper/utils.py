from itertools import chain, combinations
import math
import numpy as np


def linear_PA_probs(target_dict):
    lengths = {key: len(value) for key, value in target_dict.items()}
    denominator = sum(lengths.values())
    return [n/denominator for n in lengths.values()]


def quadratic_PA_probs(target_dict):
    lengths = {key: len(value) for key, value in target_dict.items()}
    denominator = sum(math.pow(i, 2) for i in lengths.values())
    return [math.pow(n, 2)/denominator for n in lengths.values()] 


def power_law_samples(n, min_value, max_value, exponent):
    r = np.random.uniform(0, 1, n)
    return ((max_value**(exponent+1) - min_value**(exponent+1))*r + min_value**(exponent+1))**(1/(exponent+1))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))