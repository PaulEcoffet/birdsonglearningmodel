"""Collection of functions that are day song optimisers."""

import logging
from copy import deepcopy

import numpy as np

from datasaver import QuietDataSaver
from gesture_fitter import fit_gesture_hill
from synth import gen_sound, only_sin

logger = logging.getLogger('DayOptim')


def optimise_gesture_dummy(songs, tutor_song, measure, comp, train_per_day=10,
                           nb_iter_per_train=10, datasaver=None, rng=None):
    """Optimise gestures randomly from the song models with dummy algorithm."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    for itrain in range(train_per_day):
        isong = rng.randint(len(songs))
        song = songs[isong]
        ig = rng.randint(len(song.gestures))
        start = song.gestures[ig][0]
        try:
            end = song.gestures[ig + 1][0]
        except IndexError:  # We have picked the last gesture
            end = len(tutor_song)
        logger.info('{}/{}: fit gesture {} of song {} (length {})'.format(
            itrain+1, train_per_day, ig, isong, end-start))
        prior = deepcopy(song.gestures[ig][1])
        g = measure(tutor_song[start:end])
        s = gen_sound(
            prior, end - start,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13)
        c = measure(s)
        pre_score = comp(g, c)
        res, hill_score = fit_gesture_hill(
            tutor_song[start:end].copy(), measure, comp, start_prior=prior,
            nb_iter=nb_iter_per_train, temp=None, rng=rng)
        # datasaver.add(pre_score=pre_score,
        #               new_score=hill_score, isong=isong, ig=ig)
        songs[isong].gestures[ig][1] = deepcopy(res)
        assert pre_score >= hill_score
    return songs


def optimise_gesture_cmaes(songs, tutor_song, measure, comp):
    """Optimise gestures guided with a CMA-ES algorithm."""
    raise NotImplemented
