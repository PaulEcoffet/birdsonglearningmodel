import numpy as np


def hill_climbing(function, goal, guess, guess_deviation=0.01, goal_delta=0.01, temp_max=5, max_iter=100000, seed=None, verbose=False):
    """
    Do a hill climbing algorithm to find which is the best value x so that
    function(x) = goal

    It is a Simulated Annealing algorithm to avoid getting stuck in local max

    Returns best guess and best result
    """
    if np.any(guess_deviation < 0):
        raise ValueError("The guess deviation should be a non zero positive number")
    if goal_delta < 0:
        raise ValueError("The goal error margin (goal_delta) should be a positive number")
    if max_iter < 1:
        raise ValueError("The max number of iteration without progress should be at least 1")
    size = guess.size
    best_res = function(guess)
    best_guess = guess
    best_score = np.inf

    if max_iter is None:
        max_iter = np.inf

    rng = np.random.RandomState(seed)

    i = 0
    while np.linalg.norm(goal - best_res, 2) > goal_delta and i < max_iter:
        cur_guess = rng.multivariate_normal(best_guess, guess_deviation)
        cur_res = function(cur_guess)
        if goal.shape == cur_res.shape:
            cur_score = np.linalg.norm(goal - cur_res, 2)
            temp = temp_max - temp_max*(i/max_iter)
            if cur_score < best_score or rng.uniform() < np.exp(-(cur_score - best_score)/temp):
                best_score = cur_score
                best_res = cur_res
                best_guess = cur_guess
                if verbose:
                    print(best_score, '(', i, ')')
        i += 1
    return best_guess, best_res, best_score
