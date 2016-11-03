import numpy as np
from synth import synthesize, gen_sound, dat2wav
from python_speech_features import mfcc
from hill_climbing import hill_climbing

def fit_syllable(syllable, samplerate=44100):
    best_score = np.inf
    best = None
    goal = mfcc(syllable, samplerate=samplerate, numcep=6)[..., 1:]
    for i in range(5):
        prior = np.array([
            0,
            5,
            0,
            50,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ])
        guess = np.random.multivariate_normal(prior, np.diag((2, 5, 2, 10, 2, 2, 2, 2, 2, 2, 2)))
        print("*"*40)
        print(guess)
        x, y, score = hill_climbing(
            lambda x: mfcc(gen_sound(x, len(syllable), samplerate),
                           samplerate, numcep=6)[..., 1:],
            goal, guess, guess_deviation=np.diag((1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1)),
            max_iter=20000)
        print(x)
        if score < best_score:
            best_score = score
            best = x
    print(best)

    return best


if __name__ == "__main__":
    from scipy.io import wavfile
    sr, tutor_syllable = wavfile.read('../data/ba_syllable_a.wav')
    x = fit_syllable(tutor_syllable, samplerate=sr)

    wavfile.write('out.wav', sr, dat2wav(gen_sound(x, len(tutor_syllable), sr)))
