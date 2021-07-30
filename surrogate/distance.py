import numpy as np

def simple_match_distance(x, y):
    count = 0
    for xi, yi in zip(x, y):
        if xi == yi:
            count += 1
    sim_ratio = 1.0 * count / len(x)
    return 1.0 - sim_ratio


def normalized_euclidean_distance(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))