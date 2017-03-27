"""Fit a gesture to real sound."""
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from hill_climbing import hill_climbing
from synth import gen_sound, only_sin, gen_alphabeta, synthesize


sns.set_palette('colorblind')
sns.set_context('talk')


def _get_defaults_min_max_dev(j=3):
    dev = []
    mins = []
    maxs = []
    for k in range(1, j):  # prior on sin
        dev.extend([0.05/k, 0.01/k, 0.005, 1])
        mins.extend([-50, 0, -np.pi, 0])
        maxs.extend([50, 4, np.pi, 40000])
    # last sin prior
    dev.extend([0.001, 0.001, 0.005, 100])
    mins.extend([-50, 0, -np.pi, 0])
    maxs.extend([50, 4, np.pi, 40000])
    mins.append(-5)
    maxs.append(10)
    dev.append(0.005)

    # beta
    dev.extend([0.05, 0.01, 0.05, 1, 0.005])
    mins.extend([-50, 0, -np.pi, 0, -3])
    maxs.extend([50, 3, np.pi, 1000, 2])
    return mins, maxs, dev


def fit_gesture_hill(gesture, measure, comp, prior=None, nb_iter=300,
                     temp=10, rng=None):
    """Find the parameters to fit to a gesture."""
    size = len(gesture)
    goal = measure(gesture)
    mins, maxs, dev = _get_defaults_min_max_dev()
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


def _padded_gen_sound(songmodel, range_, change_index, param, out_ab=False):
    alpha_betas = []
    for i in range_[:-1]:
        if i != change_index:
            alpha_betas.append(songmodel.gen_alphabeta(range_=[i], pad=False))
        else:
            start = songmodel.gestures[i][0]
            end = songmodel.gesture_end(i)
            size = end - start
            alpha_betas.append(gen_alphabeta(
                param, size,
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13, pad=False, beg=start))
    # last one with padding
    i = range_[-1]
    if i != change_index:
        alpha_betas.append(songmodel.gen_alphabeta(range_=[i], pad=True))
    else:
        start = songmodel.gestures[i][0]
        end = songmodel.gesture_end(i)
        size = end - start
        alpha_betas.append(gen_alphabeta(
            param, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13, pad=True, beg=start))
    if out_ab:
        return synthesize(np.concatenate(alpha_betas)), np.concatenate(alpha_betas)
    else:
        return synthesize(np.concatenate(alpha_betas))


def fit_gesture_padded(tutor, songmodel, gesture_index, measure, comp, nb_iter,
                       temp=None, rng=None):
    prev_igest = max(0, gesture_index - 1)
    start_tutor = songmodel.gestures[prev_igest][0]
    next_igest = min(len(songmodel.gestures) - 1, gesture_index + 1)
    end_tutor = songmodel.gesture_end(next_igest)
    goal = measure(tutor[start_tutor:end_tutor])
    mins, maxs, dev = _get_defaults_min_max_dev()
    sound, ab = _padded_gen_sound(
        songmodel,
        range(prev_igest, next_igest+1),
        None,
        None, out_ab=True)
    x, dummy_y, score = hill_climbing(
        function=lambda x: measure(_padded_gen_sound(
            songmodel,
            range(prev_igest, next_igest+1),
            gesture_index,
            x)),
        goal=goal,
        guess=deepcopy(songmodel.gestures[gesture_index][1]),
        guess_min=mins,
        guess_max=maxs,
        guess_deviation=np.diag(dev),
        max_iter=nb_iter,
        comparison_method=comp,
        temp_max=temp,
        verbose=False,
        rng=rng)
    sound, ab = _padded_gen_sound(
        songmodel,
        range(prev_igest, next_igest+1),
        gesture_index,
        x, out_ab=True)
    return x, score


def fit_gesture_whole(measured_tutor, songmodel, gesture_index, measure, comp,
                      nb_iter, temp=None, rng=None):
    goal = measured_tutor
    mins, maxs, dev = _get_defaults_min_max_dev()
    x, dummy_y, score = hill_climbing(
        function=lambda x: measure(_padded_gen_sound(
            songmodel,
            range(0, len(songmodel.gestures)),
            gesture_index,
            x)),
        goal=goal,
        guess=deepcopy(songmodel.gestures[gesture_index][1]),
        guess_min=mins,
        guess_max=maxs,
        guess_deviation=np.diag(dev),
        max_iter=nb_iter,
        comparison_method=comp,
        temp_max=temp,
        verbose=False,
        rng=rng)
    return x, score
