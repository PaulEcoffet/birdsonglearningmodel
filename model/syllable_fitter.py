import numpy as np
from synth import synthesize, gen_sound, dat2wav
from python_speech_features import mfcc
from hill_climbing import hill_climbing

def fit_syllable(syllable, samplerate=44100):
    best_score = np.inf
    best = None
    goal = mfcc(syllable, samplerate=samplerate, numcep=8)[..., 1:]
    #
    i = 1
    j = 3
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
    prior.extend([0, 1, 100, -1]) # beta prior
    dev.extend([0.005, 0.05, 5, 0.05])
    x, y, score = hill_climbing(
        lambda x: mfcc(gen_sound(x, len(syllable)),
                       samplerate, numcep=8)[..., 1:],
        goal, np.array(prior), guess_deviation=np.diag(dev),
        max_iter=20000, verbose=True)
    print(x)
    if score < best_score:
        best_score = score
        best = x
    print(score)
    print(best)

    return best


if __name__ == "__main__":
    from scipy.io import wavfile
    sr, tutor_syllable = wavfile.read('../data/ba_syllable_a.wav')
    x = fit_syllable(tutor_syllable, samplerate=sr)

    wavfile.write('out.wav', sr, dat2wav(gen_sound(x, len(tutor_syllable))))
