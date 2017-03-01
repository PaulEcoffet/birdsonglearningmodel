"""
Fit a whole song.

This module fits a whole song!
"""

import numpy as np
from copy import deepcopy
from fastdtw import fastdtw

from gesture_fitter import fit_gesture, _calc_res
from synth import gen_sound, only_sin

rng = np.random.RandomState()


def default_priors(nb_sin=3):
    """Give the default priors for a gesture fit."""
    prior = []
    for k in range(1, nb_sin + 1):  # prior on sin
            prior.extend([1/k, 3/k, np.pi/(k**2), 10*3**k])
    prior.append(4)
    prior.extend([0, 0, 0, 0, -0.002])  # beta prior
    return np.array(prior)


def rank(array):
    """Give rank of an array.

    [3 5 2 6]
    [2 3 1 4]
    Indeed, 2 is the 1st element of the array
    """
    temp = np.argsort(array)
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    return ranks


class SongModel:
    """Song model structure."""

    def __init__(self, gestures=None):
        """
        Initialize the song model structure.

        GTE - list of the GTE of the song
        priors - list of the priors of the song for a gesture
        """
        if gestures is None:
            gestures = [[0, default_priors()]]
        self.gestures = gestures

    def mutate(self):
        """Give a new song model with new GTEs."""
        act = rng.uniform()
        gestures = deepcopy(self.gestures)
        if act < 0.1 and len(gestures) > 2:
            to_del = rng.randint(1, len(gestures))
            del gestures[to_del]
        if act < 0.2:
            to_add = rng.randint(0, gestures[-1])
            gestures.append([to_add, default_priors()])
        else:
            to_move = rng.randint(1, len(gestures))
            gestures[to_move][0] += rng.randint(-500, 500)
        # clean GTEs
        gestures.sort(key=lambda x: x[0])
        for i in range(1, len(gestures) - 1):
            if gestures[i][0] - gestures[i - 1][0] < 400:
                del gestures[i]
        return SongModel(gestures)

    def gen_sound(self, length):
        """Generate the full song."""
        sounds = []
        for i, gesture in enumerate(self.gestures):
            start = gesture[0]
            param = gesture[1]
            try:
                end = self.gestures[i+1][0]
            except IndexError:
                end = length
            size = end - start
            sounds.append(gen_sound(
                param, size,
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13))
        out = np.stack(sounds)
        return out


def fit_song(tutor_song, sr, train_per_day=1, nb_day=2, nb_conc_song=10):
    """Fit a song with a day and a night phase."""
    songs = [SongModel() for i in range(nb_conc_song)]

    for iday in range(nb_day):
        #######
        # Day #
        #######
        for itrain in range(train_per_day):
            song = rng.choice(songs)
            ig = rng.randint(len(song.gestures))
            start = song.gestures[ig][0]
            try:
                end = song.gestures[ig + 1][0]
            except IndexError:  # We have picked the last gesture
                end = len(tutor_song)
            print('gen a song of length {}'.format(end-start))
            res, score = fit_gesture(
                tutor_song[start:end], start_prior=song.gestures[ig][1],
                nb_iter=10)
            song.gestures[ig][1] = res

        #########
        # Night #
        #########
        score = np.zeros(nb_conc_song)
        g = _calc_res(tutor_song, sr)
        for i, song in enumerate(songs):
            sig = song.gen_sound(len(tutor_song))
            c = _calc_res(sig, sr)
            score[i] = fastdtw(g, c, radius=3, dist=2)[0]
        nsongs = np.random.choice(songs, size=nb_conc_song,
                                  p=np.linalg.norm(rank(score)))
        songs = [song.mutate() for song in nsongs]
    return songs


if __name__ == '__main__':
    from scipy.io import wavfile
    sr, tsong = wavfile.read('../../data/bells.wav')
    print(fit_song(tsong, sr))
