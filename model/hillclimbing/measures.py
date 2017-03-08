"""Measures for comparing songs and song parts."""

import numpy as np
import birdsonganalysis as bsa
from python_speech_features import mfcc
from fastdtw import fastdtw


def bsa_measure(sig, sr, coefs=None):
    """Measure the song or song part with standard birdsong analysis."""
    out = []
    fnames = ['fm', 'am', 'entropy', 'goodness', 'amplitude', 'pitch']
    if coefs is None:
        coefs = {'fm': 1, 'am': 1, 'entropy': 1, 'goodness': 1,
                 'amplitude': 50, 'pitch': 1}
    features = bsa.normalize_features(
        bsa.all_song_features(sig, sr, 256, 40, 1024))
    for key in fnames:
        coef = coefs[key]
        feat = features[key]  # Verbose affectation to catch rare error
        out.append(coef * feat)
    return np.array(out).T


def mfcc_measure(sig, sr):
    """Measure the song or song part with mfcc."""
    out = mfcc(sig, sr, numcep=8, appendEnergy=True, winstep=40/sr,
               winlen=1024/sr)  # FIXME should not be hardwritten 40, 1024
    out[:, 0] = bsa.song_amplitude(
        sig, fft_step=40, fft_size=1024)[:out.shape[0]]
    return out


def get_scores(tutor_song, song_models, sr):
    """
    Get the score of each model compared to the tutor song.

    tutor_song - The signal of the tutor song
    song_models - A list of all the song models
    """
    g = bsa_measure(tutor_song, sr)
    scores = np.zeros(len(song_models))

    for i in range(len(song_models)):
        sig = song_models[i].gen_sound()
        c = bsa_measure(sig, sr)
        scores[i] = fastdtw(g, c, radius=3, dist=2)[0]
    return scores
