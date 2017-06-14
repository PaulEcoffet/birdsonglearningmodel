"""Collection of functions to optimise songs during the night.

These algorithms are mainly restructuring algorithms.
"""
import logging

import numpy as np

from datasaver import QuietDataSaver
from measures import get_scores, genetic_neighbours

logger = logging.getLogger('night_optimisers')


def rank(array):
    """Give the rank of each element of an array.

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
    nsongs = night_songs[isongs].tolist()
    datasaver.add(prev_songs=songs, prev_scores=pscore, new_songs=nsongs,
                  new_scores=score[isongs])
    return nsongs


def mutate_best_models_elite(songs, tutor_song, conf,
                             datasaver=None):
    """Elite selection and mutation of the best models.

    Keep the best mutations after each replay, parents are present in the
    selection.
    """
    if datasaver is None:
        datasaver = QuietDataSaver()
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_replay = conf['replay']
    rng = conf['rng_obj']
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
        night_songs = night_songs[isongs]
        score = score[isongs]

    nsongs = night_songs.tolist()
    datasaver.add(prev_songs=songs, prev_scores=pscore, new_songs=nsongs,
                  new_scores=score)
    return nsongs


def mutate_microbial(songs, tutor_song, conf, datasaver=None):
    """Microbial GA implementation for the songs."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    songs = np.asarray(songs)
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    for i in range(nb_replay):
        picked_songs = rng.choice(len(songs), size=2, replace=False)
        scores = get_scores(tutor_song, songs[picked_songs], measure, comp)
        best = np.argmin(scores)
        loser = 1 - best  # if best = 0, loser = 1, else: loser = 0
        songs[picked_songs[loser]] = songs[picked_songs[best]].mutate()
    return songs


def mutate_microbial_diversity(songs, tutor_song, conf, datasaver=None):
    """Microbial GA implementation for the songs."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    songs = np.asarray(songs)
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    nb_replay = conf['replay']
    rng = conf['rng_obj']
    threshold = conf.get('diversity_threshold', 2000)
    bloat_weight = conf.get('bloat_weight', 0)
    diversity_weight = conf.get('diversity_weight', 1)
    for i in range(nb_replay):
        picked_songs = rng.choice(len(songs), size=2, replace=False)
        if diversity_weight >= 10:  # Do not compute score when useless
            scores = np.array([1, 1])
        else:
            scores = get_scores(tutor_song, songs[picked_songs], measure, comp)
        nb_similar = genetic_neighbours(songs[picked_songs], songs, threshold)
        nb_gestures = np.array([len(songs[picked_songs[0]].gestures),
                                len(songs[picked_songs[1]].gestures)])
        best = np.argmin(scores * (nb_similar**diversity_weight)
                         * (nb_gestures**bloat_weight))
        loser = 1 - best  # if best = 0, loser = 1, else: loser = 0
        songs[picked_songs[loser]] = songs[picked_songs[best]].mutate()
    return songs


def extend_pop(songs, tutor_song, conf, datasaver=None):
    """Extend the size of a population."""
    if datasaver is None:
        datasaver = QuietDataSaver()
    new_pop_size = conf['night_concurrent']
    rng = conf['rng_obj']
    songs = np.asarray(songs)
    night_pop = rng.choice(songs, size=new_pop_size, replace=True)
    night_pop = np.array([song.mutate() for song in night_pop])
    return night_pop


def restrict_pop_elite(songs, tutor_song, conf, datasaver=None):
    """Restrict the size of a population with elitism (bests are kept)."""
    nb_concurrent = conf['concurrent']
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    indices = np.argpartition(get_scores(tutor_song, songs, measure, comp),
                              -nb_concurrent)[-nb_concurrent:]
    return np.asarray(songs[indices])


def restrict_pop_uniform(songs, conf, datasaver=None):
    """Restrict the size of a population by uniform random selection."""
    nb_concurrent = conf['concurrent']
    rng = conf['rng_obj']
    return rng.choice(songs, nb_concurrent, replace=False)


def restrict_pop_rank(songs, tutor_song, conf, datasaver=None):
    """Restrict the size of a population by rank driven random selection."""
    nb_concurrent = conf['concurrent']
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    rng = conf['rng_obj']
    scores = get_scores(tutor_song, night_songs, measure, comp)
    fitness = len(night_songs) - rank(score)
    return rng.choice(songs, nb_concurrent, replace=False,
                      p=fitness/np.sum(fitness))


def mutate_microbial_extended_elite(songs, tutor_song, conf, datasaver=None):
    """Do a microbial on an extended population and restrict with elitism."""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, tutor_song, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial(new_pop, tutor_song, conf, datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_elite(mutate_pop, tutor_song, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop


def mutate_microbial_extended_uniform(songs, tutor_song, conf, datasaver=None):
    """Do a microbial on an extended population and restrict by random."""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, tutor_song, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial(new_pop, tutor_song, conf, datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_uniform(mutate_pop, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop


def mutate_microbial_diversity_uniform(songs, tutor_song, conf,
                                       datasaver=None):
    """Do a microbial on an extended population and restrict by random."""
    datasaver.add(label='night', cond='before_evening', pop=songs)
    new_pop = extend_pop(songs, tutor_song, conf, datasaver)
    datasaver.add(label='night', cond='evening', pop=new_pop)
    mutate_pop = mutate_microbial_diversity(new_pop, tutor_song, conf,
                                            datasaver)
    datasaver.add(label='night', cond="before_morning", pop=mutate_pop)
    new_pop = restrict_pop_uniform(mutate_pop, conf, datasaver)
    datasaver.add(label='night', cond='morning', pop=new_pop)
    return new_pop
