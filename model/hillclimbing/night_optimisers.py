"""
Collection of functions to optimise songs during the night.

These algorithms are mainly restructuring algorithms.
"""
import logging
from copy import deepcopy

import numpy as np

from datasaver import QuietDataSaver
from measures import get_scores

logger = logging.getLogger('night_optimisers')


def rank(array):
    """
    Give the rank of each element of an array.

    >>> rank([3 5 2 6])
    [2 3 1 4]
    Indeed, 2 is the 1st smallest element of the array, 3 is the 2nd smallest,
    and so on.
    """
    temp = np.argsort(array)
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(1, len(array) + 1)
    return ranks


def mutate_best_models_dummy(songs, tutor_song, measure, comp, nb_replay,
                             datasaver=None, rng=None):
    """Dummy selection and mutation of the best models."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    nb_conc_song = len(songs)
    night_songs = np.array(songs)
    pscore = get_scores(tutor_song, songs, measure, comp)
    nb_conc_night = nb_conc_song * 2
    fitness = len(night_songs) - rank(pscore)
    night_songs = np.random.choice(night_songs, size=nb_conc_night,
                                   p=fitness/np.sum(fitness))
    night_songs = np.array([song.mutate(nb_replay) for song in night_songs])
    score = get_scores(tutor_song, night_songs, measure, comp)
    fitness = len(night_songs) - rank(score)
    isongs = rng.choice(len(night_songs),
                        size=nb_conc_song, replace=False,
                        p=fitness/np.sum(fitness))
    nsongs = deepcopy(night_songs[isongs]).tolist()
    datasaver.add(prev_songs=songs, prev_scores=pscore, new_songs=nsongs,
                  new_scores=score[isongs])
    return nsongs


def mutate_best_models_elite(songs, tutor_song, measure, comp, nb_replay,
                             datasaver=None, rng=None):
    """
    Elite selection and mutation of the best models.

    Keep the best mutations after each replay, parents are present in the
    selection.
    """
    if datasaver is None:
        datasaver = QuietDataSaver()
    if rng is None:
        rng = np.random.RandomState()
    nb_conc_song = len(songs)
    pscore = get_scores(tutor_song, songs, measure, comp)
    score = pscore
    nb_conc_night = nb_conc_song * 2
    # make night_songs an array to do list indexes.
    night_songs = np.array(songs)
    for dummy_i in range(nb_replay):
        fitness = len(night_songs) - rank(score)
        night_songs = np.random.choice(night_songs, size=nb_conc_night,
                                       p=fitness/np.sum(fitness))
        night_songs = np.array([song.mutate() for song in night_songs])
        score = get_scores(tutor_song, night_songs, measure, comp)
        fitness = len(night_songs) - rank(score)
        isongs = rng.choice(len(night_songs),
                            size=nb_conc_song, replace=False,
                            p=fitness/np.sum(fitness))
        night_songs = np.concatenate((night_songs[isongs], songs))
    nsongs = night_songs[isongs].tolist()
    datasaver.add(prev_songs=songs, prev_scores=pscore, new_songs=nsongs,
                  new_scores=score[isongs])
    return nsongs
