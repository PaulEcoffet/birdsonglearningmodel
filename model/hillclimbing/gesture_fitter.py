"""Fit a gesture to real sound."""
import numpy as np
from synth import gen_sound
from hill_climbing import hill_climbing
from fastdtw import fastdtw

import birdsonganalysis as bsa


def _calc_res(sig, sr):
    fnames = ['fm', 'pitch', 'goodness', 'entropy']
    features = bsa.normalize_features(
        bsa.all_song_features(sig, sr, without="amplitude"))
    out = []
    for key in fnames:
        out.append(features[key])
    return np.array(out).T


def fit_gesture(gesture, samplerate=44100):
    """Find the parameters to fit to a gesture."""
    size = len(gesture)
    #goal = _calc_res(gesture, samplerate)
    goal = np.copy(gesture)
    i = 1
    j = 3
    prior = []
    dev = []
    mins = []
    maxs = []
    for k in range(1, i + 1):  # prior on exponential
        prior.extend([0.01/k, 10/k])
        dev.extend([0.05, 0.05])
        mins.extend([-10, 0])
        maxs.extend([10, 5000])
    for k in range(1, j + 1):  # prior on sin
        prior.extend([0.5/k, 0, 200*k**2])
        dev.extend([0.01/k, 0.005, 5*k])
        mins.extend([0, -np.pi, 0])
        maxs.extend([3, np.pi, 8000])
    prior.append(4)
    mins.append(0)
    maxs.append(10)
    dev.append(0.5)

    # hard prior
    prior = [-10, 2733.9, 1.69, 0.59, 30, 1.39, 0.29, 610, 0.24, -1.41,
             3020, 5.8]

    prior.extend([0, 0, 0, 0, 1, -0.006])  # beta prior
    dev.extend([0.00, 0.00, 0.00, 0.00, 0, 0.000])
    mins.extend([-10, 0, 0, -np.pi, 0, -3])
    maxs.extend([10, 1000, 3, np.pi, 1000, 0])
    x, y, score = hill_climbing(
        # function=lambda x: _calc_res(gen_sound(x, size), samplerate),
        function=lambda x: gen_sound(x, size),
        goal=goal,
        guess=np.array(prior),
        guess_min=mins,
        guess_max=maxs,
        guess_deviation=np.diag(dev),
        max_iter=5000,
        comparison_method=lambda g, c: np.linalg.norm(g - c),
        temp_max=100,
        verbose=False)
    return x, score


if __name__ == "__main__":
    from scipy.io import wavfile
    import os
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('paper')
    from synth import f_str
    from synth import gen_alphabeta
    sr, tutor_syllable = wavfile.read('../../data/ba_syllable_a_end.wav')
    sr, synth_syllable = wavfile.read('../../data/ba_syllable_a_end_out.wav')
    best_x = None
    best_score = np.inf
    run_name = datetime.datetime.now().strftime('%y%m%d_%H%M')
    os.makedirs('res/{}/'.format(run_name), exist_ok=True)
    plt.figure(figsize=(16, 5))
    plt.plot(tutor_syllable)
    plt.savefig('res/{}/ref_song.svg'.format(run_name))
    plt.figure(figsize=(16, 5))
    true_params = np.loadtxt('../../data/ba_syllable_a_end_ab.dat')
    plt.plot(true_params)
    plt.savefig('res/{}/ref_params.svg'.format(run_name))
    #g = _calc_res(tutor_syllable, sr)
    #c = _calc_res(synth_syllable, sr)
    g = tutor_syllable
    c = synth_syllable
    #score = fastdtw(g, c)[0]
    score = np.linalg.norm(g - c)
    print('score between real and synth: {}'.format(score))
    for i in range(1):
        try:
            x, score = fit_gesture(tutor_syllable, samplerate=sr)
            print("*"*80)
            plt.figure(figsize=(16, 5))
            plt.plot(gen_sound(x, len(tutor_syllable)))
            plt.savefig('res/{}/{}_song.svg'.format(run_name, i))
            print(x)
            print('alpha(t) = ' + f_str(x[:1*2+3*3+1],
                                        nb_exp=1, nb_sin=3, x='t'))
            print('beta(t) = ' + f_str(x[1*2+3*3+1:],
                                       nb_exp=1, nb_sin=1, x='t'))
            alpha_beta = gen_alphabeta(x, len(tutor_syllable))
            plt.figure(figsize=(16, 5))
            plt.plot(np.arange(0, len(tutor_syllable) + 2) / 44100,
                     alpha_beta[:, 0], label='a')
            plt.plot(np.arange(0, len(tutor_syllable) + 2) / 44100,
                     alpha_beta[:, 1], label='b')
            plt.legend()
            plt.savefig('res/{}/{}_param.svg'.format(run_name, i))
            if score < best_score:
                best_x = x
                best_score = score
            print('{}: {} (best: {})'.format(i, score, best_score))
            wavfile.write('res/{}/out_{}_{}.wav'.format(run_name, i, score),
                          sr, gen_sound(x, len(tutor_syllable)))
        except KeyboardInterrupt:
            break

    print(best_x)
    print('score', best_score)
    wavfile.write('out.wav', sr, gen_sound(best_x,
                                           len(tutor_syllable)))
