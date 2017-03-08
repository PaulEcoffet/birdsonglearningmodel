"""
Collection of functions to optimise songs during the night.

These algorithms are mainly restructuring algorithms.
"""
import numpy as np
from measures import get_scores
import logging
from copy import deepcopy

logger = logging.getLogger('night_optimisers')

rng = np.random.RandomState()


def rank(array):
    """Give rank of an array.

    [3 5 2 6]
    [2 3 1 4]
    Indeed, 2 is the 1st smallest element of the array
    """
    temp = np.argsort(array)
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(1, len(array) + 1)
    return ranks


def mutate_best_models_dummy(songs, tutor_song, measure, comp, nb_replay):
    """Dummy selection and mutation of the best models."""
    nb_conc_song = len(songs)
    night_songs = np.array(songs)
    score = get_scores(tutor_song, songs, measure, comp)
    nb_conc_night = nb_conc_song * 2
    fitness = len(night_songs) - rank(score)
    night_songs = np.random.choice(night_songs, size=nb_conc_night,
                                   p=fitness/np.sum(fitness))
    for ireplay in range(nb_replay):
        logger.info('mutation {} out of {}'.format(ireplay, nb_replay))
        night_songs = np.array([song.mutate() for song in night_songs])
    score = get_scores(tutor_song, night_songs, measure, comp)
    fitness = (len(night_songs)) - rank(score)
    isongs = rng.choice(len(night_songs),
                        size=nb_conc_song, replace=False,
                        p=fitness/np.sum(fitness))
    logger.debug(score[isongs])
    songs = deepcopy(night_songs[isongs])
    songs = songs.tolist()
    return songs
