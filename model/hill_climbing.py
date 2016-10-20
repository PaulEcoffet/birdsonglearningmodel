import numpy as np


def hill_climbing(function, goal, guess, guess_deviation=0.01, goal_delta=0.01, max_iter=100000, seed=None):
    if guess_deviation <= 0:
        raise ValueError("The guess deviation should be a non zero positive number")
    if goal_delta < 0:
        raise ValueError("The goal error margin (goal_delta) should be a positive number")
    if max_iter < 1:
        raise ValueError("The max number of iteration without progress should be at least 1")
    size = guess.size
    best_res = function(guess)
    if best_res.size != goal.size:
        raise ValueError("output of the function not of the same dimension as the goal")
    best_guess = guess

    if max_iter is None:
        max_iter = np.inf

    rng = np.random.RandomState(seed)

    i = 0
    while np.linalg.norm(goal - best_res, 2) > goal_delta and i < max_iter:
        cur_guess = best_guess + rng.normal(0, guess_deviation, size)
        cur_res = function(cur_guess)
        if np.linalg.norm(goal - cur_res, 2) < np.linalg.norm(goal - best_res, 2):
            best_res = cur_res
            best_guess = cur_guess
            i = 0
        else:
            i += 1
    return best_guess, best_res
