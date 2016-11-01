import numpy as np
from synth import synthesize, gen_sound, dat2wav
from python_speech_features import mfcc
from hill_climbing import hill_climbing

def fit_syllable(syllable, samplerate=44100):
    best_score = np.inf
    best = None
    goal = mfcc(syllable, samplerate=samplerate, winlen=0.01, winstep=0.002)[..., 1:]
    for i in range(40):
        prior = [50, # alpha, a*t
                 1400, # constant, on average, if alpha non zero, E(a) = 1400
                 200, # c*cos, assumed to be big
                 0, # period and phase are short
                 0, # must stay between 0..2pi
                 0, # beta, very small
                 -0.10, # mean is -0.10 for b
                 0, # c*cos
                 0, # period
                 0] # phase
        guess = np.random.multivariate_normal(prior, np.diag((20, 200, 20, 2, 2, 2, 2, 2, 2, 2)))
        print("*"*40)
        print(guess)
        x, y, score = hill_climbing(
            lambda x: mfcc(gen_sound(x, len(syllable), samplerate),
                           samplerate,
                           winlen=0.01, winstep=0.002)[..., 1:],
            goal, guess, guess_deviation=np.diag((10, 50, 5, 0.5, 0.5, 0.1, 0.2, 0.2, 0.2, 0.1)),
            max_iter=20000)
        print(x)
        if score < best_score:
            best_score = score
            best = x
    print(best)

    return best


if __name__ == "__main__":
    from scipy.io import wavfile
    sr, tutor_syllable = wavfile.read('../data/ba_syllable_b.wav')
    x = fit_syllable(tutor_syllable, samplerate=sr)

    wavfile.write('out.wav', sr, dat2wav(gen_sound(x, len(tutor_syllable), sr)))
