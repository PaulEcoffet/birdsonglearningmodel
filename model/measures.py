"""Measures for comparing songs and song parts."""

import numpy as np
import birdsonganalysis as bsa
from python_speech_features import mfcc


def bsa_measure(sig, sr, coefs=None):
    """Measure the song or song part with standard birdsong analysis."""
    out = []
    fnames = ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch']
    if coefs is None:
        coefs = {'fm': 1, 'am': 50, 'entropy': 1, 'goodness': 1,
                 'amplitude': 50, 'pitch': 1}
    features = bsa.normalize_features(
        bsa.all_song_features(sig, sr, 256, 40, 1024))
    for key in fnames:
        coef = coefs[key]
        feat = features[key]  # Verbose affectation to catch rare error
        out.append(coef * feat)
    out = np.array(out).T
    return out


def mfcc_measure(sig, sr):
    """Measure the song or song part with mfcc."""
    out = mfcc(sig, sr, numcep=8, appendEnergy=True, winstep=40/sr,
               winlen=1024/sr)  # FIXME should not be hardwritten 40, 1024
    out[:, 0] = bsa.song_amplitude(
        sig, fft_step=40, fft_size=1024)[:out.shape[0]]
    return out


def get_scores(tutor_song, song_models, measure, comp):
    """
    Get the score of each model compared to the tutor song.

    tutor_song - The signal of the tutor song
    song_models - A list of all the song models
    """
    g = measure(tutor_song)
    scores = np.zeros(len(song_models))
    for i in range(len(song_models)):
        sig = song_models[i].gen_sound()
        c = measure(sig)
        scores[i] = comp(g, c)
    return scores
