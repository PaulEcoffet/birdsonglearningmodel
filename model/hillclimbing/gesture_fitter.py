"""Fit a gesture to real sound."""
import numpy as np
from synth import gen_sound, only_sin
from hill_climbing import hill_climbing
from fastdtw import fastdtw

import birdsonganalysis as bsa


def _calc_res(sig, sr):
    out = []
    fnames = ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch']
    features = bsa.normalize_features(
        bsa.all_song_features(sig, sr, 256, 40, 1024))
    for key in fnames:
        if key == 'amplitude':
            coef = 100
        else:
            coef = 1
        out.append(coef * features[key])
    return np.array(out).T


def fit_gesture(gesture, samplerate=44100):
    """Find the parameters to fit to a gesture."""
    size = len(gesture)
    goal = _calc_res(gesture, samplerate)
    j = 3
    prior = []
    dev = []
    mins = []
    maxs = []
    for k in range(1, j + 1):  # prior on sin
        prior.extend([0, 3/k, np.pi/(k**2), 2**k])
        dev.extend([1/k, 0.01/k, 0.005, 5*k])
        mins.extend([-100, 0, -np.pi, 0])
        maxs.extend([100, 3, np.pi, 8000])
    prior.append(4)
    mins.append(0)
    maxs.append(10)
    dev.append(0.5)

    # hard prior
    # prior = [0, 3.21, 1.6, 15, -62.59, 1.64, 1.2, 600, 2.87, 0.52, 2.5, 1200,
    #         4.9]

    prior.extend([0, 0, 0, 0, -0.006])  # beta prior
    dev.extend([0.00, 0.00, 0.00, 0, 0.000])
    mins.extend([-100, 0, -np.pi, 0, -3])
    maxs.extend([100, 3, np.pi, 1000, 0])
    x, y, score = hill_climbing(
        function=lambda x: _calc_res(gen_sound(
            x, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13), samplerate),
        goal=goal,
        guess=np.array(prior),
        guess_min=mins,
        guess_max=maxs,
        guess_deviation=np.diag(dev),
        max_iter=10000,
        comparison_method=lambda g, c: fastdtw(g, c, dist=2, radius=3)[0],
        temp_max=50,
        verbose=False)
    return x, score


if __name__ == "__main__":
    from scipy.io import wavfile
    import os
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('paper')
    from synth import gen_alphabeta

    sr, tutor_syllable = wavfile.read('../../data/ba_syllable_a.wav')
    sr, synth_syllable = wavfile.read('../../data/ba_syllable_a_out.wav')
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
    g = _calc_res(tutor_syllable, sr)
    c = _calc_res(synth_syllable, sr)
    score = fastdtw(c, g, dist=2, radius=3)[0]
    print('score between real and synth: {}'.format(score))
    for i in range(1):
        try:
            x, score = fit_gesture(tutor_syllable, samplerate=sr)
            print("*"*80)
            plt.figure(figsize=(16, 5))
            plt.plot(gen_sound(x, len(tutor_syllable),
                               falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                               fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                               falpha_nb_args=13))
            plt.savefig('res/{}/{}_song.svg'.format(run_name, i))
            print(x)
            alpha_beta = gen_alphabeta(x, len(tutor_syllable),
                                       falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                                       fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                                       falpha_nb_args=13)
            plt.figure(figsize=(16, 5))
            plt.plot(alpha_beta[:, 0], label='a')
            plt.plot(alpha_beta[:, 1], label='b')
            plt.legend()
            plt.savefig('res/{}/{}_param.svg'.format(run_name, i))
            if score < best_score:
                best_x = x
                best_score = score
            print('{}: {} (best: {})'.format(i, score, best_score))
            wavfile.write('res/{}/out_{}_{}.wav'.format(run_name, i, score),
                          sr, gen_sound(x, len(tutor_syllable),
                                        falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                                        fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                                        falpha_nb_args=13))
        except KeyboardInterrupt:
            break

    print(best_x)
    print('score', best_score)
    wavfile.write('out.wav', sr, gen_sound(best_x,
                                           len(tutor_syllable),
                                           falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                                           fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                                           falpha_nb_args=13))
