import numpy as np
from synth import synthesize, gen_sound, dat2wav
from python_speech_features import mfcc
from hill_climbing import hill_climbing
from fastdtw import fastdtw


def fit_gesture(gesture, samplerate=44100):
    best_score = np.inf
    best = None
    goal = mfcc(gesture, samplerate=samplerate, winstep=0.005, numcep=26, nfilt=52, nfft=1024)[..., 1:]
    i = 1
    j = 3
    prior = []
    dev = []
    mins = []
    maxs = []
    for k in range(1,i+1):  # prior on exponential
        prior.extend([0.5/k, 10/k])
        dev.extend([0.05, 0.05])
        mins.extend([-10, 0])
        maxs.extend([10, 10])
    for k in range(1,j+1):
        prior.extend([1/k, 0, 20*k**2])
        dev.extend([0.001, 0.005, 5*k])
        mins.extend([0, -np.pi, 0])
        maxs.extend([10, np.pi, 2000])
    prior.append(0.3)
    mins.append(0)
    maxs.append(10)
    dev.append(0.005)

    prior.extend([0, 0, 0, 1, 100, -0.028]) # beta prior
    dev.extend([0.01, 0.01, 0.01, 0.01, 10, 0.001])
    mins.extend([-10, 0, 0, -np.pi, 0, -3])
    maxs.extend([10, 10, 3, np.pi, 1000, 0])
    x, y, score = hill_climbing(
        lambda x: mfcc(gen_sound(x, len(syllable), alphaf_shape=(i, j)),
                       samplerate,winstep=0.005, numcep=26, nfilt=52, nfft=1024)[..., 1:],
        goal, np.array(prior), guess_min=mins, guess_max=maxs, guess_deviation=np.diag(dev),
        max_iter=500, comparison_method=lambda g, c: fastdtw(g, c, dist=2)[0],
        temp_max=10,
        verbose=False)


    return x, score


if __name__ == "__main__":
    from scipy.io import wavfile
    import os
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from synth import f_str
    from synth import gen_alphabeta
    sr, tutor_syllable = wavfile.read('../data/ba_syllable_b.wav')
    plt.plot(tutor_syllable)
    plt.show()
    best_x = None
    best_score = np.inf
    run_name = datetime.datetime.now().strftime('%y%m%d_%H%M')
    os.makedirs('res/{}/'.format(run_name), exist_ok=True)
    for i in range(20):
        try:
            x, score = fit_syllable(tutor_syllable, samplerate=sr)
            print("*"*80)
            plt.plot(gen_sound(x, len(tutor_syllable)))
            plt.show()
            print(x)
            print('alpha(t) = ' + f_str(x[:1*2+3*3+1], nb_exp=1, nb_sin=3, x='t'))
            print('beta(t) = ' + f_str(x[1*2+3*3+1:], nb_exp=1, nb_sin=1, x='t'))
            alpha_beta = gen_alphabeta(x, len(tutor_syllable))
            plt.plot(np.arange(0,len(tutor_syllable)+2)/44100, alpha_beta[:, 0], label='a')
            plt.plot(np.arange(0,len(tutor_syllable)+2)/44100, alpha_beta[:, 1], label='b')
            plt.legend()
            plt.show()
            if score < best_score:
                best_x = x
                best_score = score
            print('{}: {} (best: {})'.format(i, score, best_score))
            wavfile.write('res/{}/out_{}_{}.wav'.format(run_name, i, score), sr, dat2wav(gen_sound(x, len(tutor_syllable))))
        except KeyboardInterrupt:
            break

    print(best_x)
    print('score', best_score)
    wavfile.write('out.wav', sr, dat2wav(gen_sound(best_x, len(tutor_syllable))))
