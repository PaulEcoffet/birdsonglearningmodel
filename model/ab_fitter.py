"""
Module that fit an equation to alpha and beta raw data
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from fastdtw import fastdtw

from hill_climbing import hill_climbing

def fit_func(goal, f, params, dev):
    """
    Find the params so that the function f fits the best the goal by
    minimizing square error, return the best params.

    argmin_{params} (f(t, params) - y(t))^2, t=0..len(goal)

    goal - An array of the value that the function must fit.
        Assumes f(i) = goal[i]
    f - The function to fit. must be of the signature f(i, params) with params
        being an ndarray.
    params - Either guesses for the params as an ndarray, or the number of params
             to fit.
    returns best_params, sq_error
    """
    try:
        nb_params = len(params)
        guess = copy.copy(params)
    except TypeError:
        nb_params = params
        guess = np.ones(nb_params)
    params_out, y, score = hill_climbing(
        lambda param: f(np.arange(len(goal)), param),
        goal, guess, guess_deviation=dev,
        comparison_method=lambda g, c: fastdtw(g, c, dist=2)[0],
        max_iter=100000)
    return params_out, score


if __name__ == "__main__":
    def f(x, p, nb_exp=2, nb_sin=2):
        ip = np.nditer(p)
        x = x/22500
        return np.sum([next(ip) * np.exp(-next(ip) * x) for i in range(nb_exp)]
                    + [next(ip) * np.sin((next(ip) + 2*np.pi * x) * next(ip)) for i in range(nb_sin)] + [next(ip)], axis=0)

    for compt in range(10):
        print('-'*40)
        print(compt)
        print('*-*')
        for syb in ['a', 'b']:
            goal = np.loadtxt('../data/ba_syllable_{}_ab.dat'.format(syb))[:, 0]
            print(goal)
            print("*"*80)
            i = 1
            j = 3
            print(i, j)
            prior = []
            dev = []
            for k in range(1,i+1):  # prior on exponential
                prior.extend([1/k, 1/k])
                dev.extend([0.05, 0.05])
            for k in range(1,j+1):
                prior.extend([10/k, 0, 50*k**2])
                dev.extend([0.1, 0.05, 5*k])
            prior.append(0)
            dev.append(0.05)
            params, score = fit_func(goal, lambda x, p: f(x, p, i, j), np.array(prior),
                                     dev=np.diag(dev))
            print(score)
            print(params)
            plt.figure(i*10 + j)
            plt.plot(np.arange(len(goal))/22500, goal)
            plt.plot(np.arange(len(goal))/22500, f(np.arange(len(goal)), params, i, j))
            plt.title('exp : {}, sin : {}'.format(i, j))
            plt.savefig('res/{}_{}.svg'.format(syb, compt), format='svg')
            plt.savefig('res/{}_{}.png'.format(syb, compt), format='png')
            plt.show()
