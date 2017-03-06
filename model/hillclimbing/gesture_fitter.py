"""Fit a gesture to real sound."""
import sys

import numpy as np
from fastdtw import fastdtw

import birdsonganalysis as bsa
from python_speech_features import mfcc
from hill_climbing import hill_climbing
from synth import gen_sound, only_sin


def _calc_res(sig, sr):
    out = []
    fnames = ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch']
    coefs = {'fm': 1, 'am': 1, 'entropy': 1, 'goodness': 1, 'amplitude': 50,
             'pitch': 1}
    features = bsa.normalize_features(
        bsa.all_song_features(sig, sr, 256, 40, 1024))
    for key in fnames:
        out.append(coefs[key] * features[key])
    return np.array(out).T


def _calc_res_(sig, sr):
    out = mfcc(sig, sr, numcep=8, appendEnergy=True, winstep=40/sr,
               winlen=1024/sr)
    out[:, 0] = bsa.song_amplitude(
        sig, fft_step=40, fft_size=1024)[:out.shape[0]]
    return out


def fit_gesture(gesture, samplerate=44100, start_prior=None, nb_iter=100):
    """Find the parameters to fit to a gesture."""
    size = len(gesture)
    goal = _calc_res(gesture, samplerate)
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
        max_iter=nb_iter,
        comparison_method=lambda g, c: fastdtw(g, c, dist=2, radius=3)[0],
        # comparison_method=lambda g, c: np.linalg.norm(g - c),
        temp_max=20,
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

    fname = 'ba_syllable_a'
    sr, tutor_syllable = wavfile.read('../../data/{}.wav'.format(fname))
    sr, synth_syllable = wavfile.read('../../data/{}_out.wav'.format(fname))
    true_params = np.loadtxt('../../data/{}_ab.dat'.format(fname))
    best_x = None
    best_score = np.inf
    run_name = datetime.datetime.now().strftime('%y%m%d_%H%M')
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
    g = _calc_res(tutor_syllable, sr)
    c = _calc_res(synth_syllable, sr)
    score = fastdtw(c, g, dist=2, radius=3)[0]
    # score = np.linalg.norm(g - c)
    print('score between real and synth: {}'.format(score))
    for i in range(1):
        try:
            x, score = fit_gesture(tutor_syllable, samplerate=sr)
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
