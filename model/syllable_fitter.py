import numpy as np
from synth import synthesize, gen_sound, dat2wav
from python_speech_features import mfcc
from hill_climbing import hill_climbing
from fastdtw import fastdtw


def fit_syllable(syllable, samplerate=44100):
    best_score = np.inf
    best = None
    goal = mfcc(syllable, samplerate=samplerate,  winlen=0.01, winstep=0.003, numcep=13)[..., 1:]
    #
    i = 1
    j = 3
    prior = []
    dev = []
    for k in range(1,i+1):  # prior on exponential
        prior.extend([0.5/k, 10/k])
        dev.extend([0.05, 0.05])
    for k in range(1,j+1):
        prior.extend([0.3/k, 0, 20*k**2])
        dev.extend([0.001, 0.005, 5*k])
    prior.append(0.3)
    dev.append(0.005)
    prior.extend([0, 0, 0, 1, 100, -0.010]) # beta prior
    dev.extend([0.005, 0.005, 0.005, 0.005, 5, 0.001])
    x, y, score = hill_climbing(
        lambda x: mfcc(gen_sound(x, len(syllable)),
                       samplerate, winlen=0.01, winstep=0.003, numcep=13)[..., 1:],
        goal, np.array(prior), guess_deviation=np.diag(dev),
        max_iter=500, comparison_method=lambda g, c: fastdtw(g, c, dist=2)[0],
        temp_max=10,
        verbose=False)


    return x, score


if __name__ == "__main__":
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    import seaborn as sns
    from synth import f_str
    from synth import gen_alphabeta
    sr, tutor_syllable = wavfile.read('../data/ba_syllable_a.wav')
    plt.plot(tutor_syllable)
    plt.show()
    best_x = None
    best_score = np.inf
    for i in range(20):
        try:
            x, score = fit_syllable(tutor_syllable, samplerate=sr)

            plt.plot(gen_sound(x, len(tutor_syllable)))
            plt.show()
            print('alpha(t) = ' + f_str(x[:1*2+3*3+1], nb_exp=1, nb_sin=3, x='t'))
            print('beta(t) = ' + f_str(x[1*2+3*3+1:], nb_exp=1, nb_sin=1, x='t'))
            plt.plot(np.arange(0,len(tutor_syllable)+2)/44100, gen_alphabeta(x, len(tutor_syllable))[:, 0], label='a')
            plt.plot(np.arange(0,len(tutor_syllable)+2)/44100, gen_alphabeta(x, len(tutor_syllable))[:, 1], label='b')
            plt.legend()
            plt.show()
            if score < best_score:
                best_x = x
                best_score = score
            print('{}: {} (best: {})'.format(i, score, best_score))
        except KeyboardInterrupt:
            break

    print(best_x)
    print('score', best_score)
    wavfile.write('out.wav', sr, dat2wav(gen_sound(best_x, len(tutor_syllable))))
