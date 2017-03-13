"""Fit a gesture to real sound."""
import sys

import numpy as np
from measures import bsa_measure
from fastdtw import fastdtw

from hill_climbing import hill_climbing
from synth import gen_sound, only_sin


def fit_gesture_hill(gesture, measure, comp, start_prior=None, nb_iter=300,
                     logger=None, temp=10, rng=None):
    """Find the parameters to fit to a gesture."""
    size = len(gesture)
    goal = measure(gesture)
    j = 3
    prior = []
    dev = []
    mins = []
    maxs = []
    for k in range(1, j + 1):  # prior on sin
        prior.extend([1/k, 3/k, np.pi/(k**2), 10*3**k])
        dev.extend([0.05/k, 0.1/k, 0.05, 5*(k**2)])
        mins.extend([-100, 0, -np.pi, 0])
        maxs.extend([100, 3, np.pi, 8000])
    prior.append(4)
    mins.append(0)
    maxs.append(10)
    dev.append(0.5)

    if start_prior is not None:
        prior = start_prior
    else:
        prior.extend([0, 0, 0, 0, -0.002])  # beta prior
    dev.extend([0.1, 0.1, 0.1, 50, 0.0001])
    mins.extend([-100, 0, -np.pi, 0, -3])
    maxs.extend([100, 3, np.pi, 1000, 2])
    x, y, score = hill_climbing(
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
        logger=logger,
        rng=rng)
    return x, score


if __name__ == "__main__":
    from scipy.io import wavfile
    import os
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('paper')
    from synth import gen_alphabeta
    import pickle

    fname = 'ba_syllable_a'
    sr, tutor_syllable = wavfile.read('../../data/{}.wav'.format(fname))
    sr, synth_syllable = wavfile.read('../../data/{}_out.wav'.format(fname))
    true_params = np.loadtxt('../../data/{}_ab.dat'.format(fname))
    best_x = None
    best_score = np.inf
    run_name = datetime.datetime.now().strftime('gesture_%y%m%d_%H%M%S')
    try:
        name = sys.argv[1]
    except IndexError:
        pass
    else:
        run_name += '_{}'.format(name)
    os.makedirs('res/{}/'.format(run_name), exist_ok=True)
    plt.figure(figsize=(16, 5))
    plt.plot(tutor_syllable)
    plt.savefig('res/{}/ref_song.svg'.format(run_name))
    plt.figure(figsize=(16, 5))
    plt.plot(true_params)
    plt.savefig('res/{}/ref_params.svg'.format(run_name))
    g = bsa_measure(tutor_syllable, sr)
    c = bsa_measure(synth_syllable, sr)
    score = fastdtw(c, g, dist=2, radius=3)[0]
    # score = np.linalg.norm(g - c)
    print('score between real and synth: {}'.format(score))
    for i in range(1):
        try:
            logger = []
            x, score = fit_gesture_hill(tutor_syllable,
                                        logger=logger)
            with open('res/{}/log.pkl'.format(run_name), 'wb') as f:
                pickle.dump(logger, f)
            print("*"*80)
            plt.figure(figsize=(16, 5))
            plt.plot(gen_sound(
                x, len(tutor_syllable),
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13))
            plt.savefig('res/{}/{}_song.svg'.format(run_name, i))
            print(x)
            alpha_beta = gen_alphabeta(
                x, len(tutor_syllable),
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13)
            fig, axs = plt.subplots(2, 1, figsize=(16, 5))
            axs[0].plot(alpha_beta[:, 0], label='a')
            axs[1].plot(alpha_beta[:, 1], color='g', label='b')
            axs[0].legend()
            axs[1].legend
            plt.savefig('res/{}/{}_param.svg'.format(run_name, i))
            if score < best_score:
                best_x = x
                best_score = score
            print('{}: {} (best: {})'.format(i, score, best_score))
            wavfile.write('res/{}/out_{}_{}.wav'.format(run_name, i, score),
                          sr,
                          gen_sound(
                              x, len(tutor_syllable),
                              falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                              fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                              falpha_nb_args=13))
        except KeyboardInterrupt:
            break

    print(best_x)
    print('score', best_score)
    wavfile.write('out.wav', sr,
                  gen_sound(
                      best_x, len(tutor_syllable),
                      falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                      fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                      falpha_nb_args=13))
