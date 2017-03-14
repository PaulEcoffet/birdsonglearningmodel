"""Fit a gesture to real sound."""
import sys

import numpy as np
from measures import bsa_measure
from fastdtw import fastdtw

from hill_climbing import hill_climbing
from synth import gen_sound, only_sin


def fit_gesture_hill(gesture, measure, comp, start_prior=None, nb_iter=300,
                     temp=10, rng=None):
    """Find the parameters to fit to a gesture."""
    size = len(gesture)
    goal = measure(gesture)
    j = 3
    prior = []
    dev = []
    mins = []
    maxs = []
    for k in range(1, j + 1):  # prior on sin
        prior.extend([0/k, 0/k, np.pi/(k**2), 10*3**k])
        dev.extend([0.005/k, 0.001/k, 0.005, 5*(k**2)])
        mins.extend([-50, 0, -np.pi, 0])
        maxs.extend([50, 2, np.pi, 8000])
    prior.append(4)
    mins.append(0)
    maxs.append(10)
    dev.append(0.5)

    if start_prior is not None:
        prior = start_prior
    else:
        prior.extend([0, 0, 0, 0, -0.002])  # beta prior
    dev.extend([0.005, 0.005, 0.005, 10, 0.0001])
    mins.extend([-100, 0, -np.pi, 0, -3])
    maxs.extend([100, 3, np.pi, 1000, 2])
    x, dummy_y, score = hill_climbing(
        function=lambda x: measure(gen_sound(
            x, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13)),
        goal=goal,
        guess=np.array(prior),
        guess_min=mins,
        guess_max=maxs,
        guess_deviation=np.diag(dev),
        max_iter=nb_iter,
        comparison_method=comp,
        temp_max=temp,
        verbose=False,
        rng=rng)
    return x, score
