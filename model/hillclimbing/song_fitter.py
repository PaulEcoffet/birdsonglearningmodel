"""
Fit a whole song.

This module fits a whole song!
"""

import argparse as ap
from copy import deepcopy

import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile

from day_optimisers import optimise_gesture_dummy
from night_optimisers import mutate_best_models_dummy
from measures import get_scores, bsa_measure
from song_model import SongModel
import logging

rng = np.random.RandomState()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('root')


def fit_song(tutor_song, measure, comp, day_optimisation, night_optimisation,
             day_conf, night_conf, nb_day=5, nb_conc_song=3, nb_split=10):
    """Fit a song with a day and a night phase."""
    songs = [SongModel(song=tutor_song, nb_split=nb_split, rng=rng)
             for i in range(nb_conc_song)]
    songlog = []
    songlog.append(('Start', songs,
                    get_scores(tutor_song, songs, measure, comp)))

    for iday in range(nb_day):
        logger.info('☀️️\t☀️️\t☀️️\tDay {} of {}\t☀️️\t☀️️\t☀️'.format(iday+1, nb_day)) # noqa
        songs = day_optimisation(deepcopy(songs),
                                 tutor_song, measure, comp, **day_conf)
        score = get_scores(tutor_song, songs, measure, comp)
        if iday + 1 != nb_day:
            logger.debug(score)
            songlog.append(('BeforeNight', deepcopy(songs), deepcopy(score)))
            logger.info('💤\t💤\t💤\tNight\t💤\t💤\t💤')
            songs = night_optimisation(deepcopy(songs),
                                       tutor_song, measure, comp,
                                       **night_conf)
            score = get_scores(tutor_song, songs, measure, comp)
            songlog.append(('AfterNight', deepcopy(songs), deepcopy(score)))
    songlog.append(('End', songs, get_scores(tutor_song, songs, measure,
                                             comp)))
    return songs, songlog


def main():
    """Main function for this module, called if not imported."""
    import os
    import datetime
    import pickle
    start = datetime.datetime.now()
    parser = ap.ArgumentParser(
        description="""
        reproduce the learning of a zebra finch for a given tutor song.
        """
    )
    parser.add_argument('tutor', type=ap.FileType('rb'),
                        help='The targeted song to learn')
    parser.add_argument('-d', '--days', type=int, default=45,
                        help='number of days')
    parser.add_argument('-t', '--train-per-day', type=int, default=100,
                        help='number of training for a gesture per day')
    parser.add_argument('-c', '--concurrent', type=int, default=3,
                        help='number of concurrent model for the song')
    parser.add_argument('-n', '--name', type=str, default='noname',
                        help='name of the trial for logging')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='seed for the random number generator')
    parser.add_argument('-r', '--replay', type=int, default=3,
                        help='number of passes for new generations during'
                        ' night')
    parser.add_argument('-i', '--iter-per-train', type=int, default=20,
                        help='number of iteration when training a gesture')
    args = parser.parse_args()
    if args.seed is None:
        seed = int(datetime.datetime.now().timestamp())
    else:
        seed = args.seed
    rng.seed(seed)
    sr, tsong = wavfile.read(args.tutor)
    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    path = 'res/{}_{}'.format(date, args.name)
    os.makedirs(path)
    wavfile.write(os.path.join(path, 'tutor.wav'), sr, tsong)
    data = {'tutor': tsong,
            'sr': sr,
            'days': args.days,
            'train_per_day': args.train_per_day,
            'concurrent': args.concurrent,
            'name': args.name,
            'seed': seed,
            'replay': args.replay,
            'iter_per_train': args.iter_per_train}
    with open(os.path.join(path, 'params.pkl'), 'wb') as f:
        pickle.dump(data, f)
    day_conf = {'nb_iter_per_train': args.iter_per_train,
                'train_per_day': args.train_per_day}
    night_conf = {'nb_replay': args.replay}
    try:
        songs, song_log = fit_song(
            tsong,
            measure=lambda x: bsa_measure(x, sr),
            comp=lambda g, c: np.linalg.norm(g - c),
            day_optimisation=optimise_gesture_dummy,
            night_optimisation=mutate_best_models_dummy,
            day_conf=day_conf,
            night_conf=night_conf,
            nb_day=args.days,
            nb_conc_song=args.concurrent,
            nb_split=30)
    except KeyboardInterrupt as e:
        with open(os.path.join(path, 'aborted.txt'), 'a') as f:
            f.write('aborted')
        raise e
    logger.info('!!!! Learning over !!!!')
    logger.info('Logging the songs')
    with open(os.path.join(path, 'songs.pkl'), 'wb') as f:
        pickle.dump(songs, f)
    with open(os.path.join(path, 'songlog.pkl'), 'wb') as f:
        pickle.dump(song_log, f)
    logger.info('Generating the waves')
    for i, song in enumerate(songs):
        wavfile.write(os.path.join(path, 'out_{}.wav'.format(i)),
                      44100, song.gen_sound())
    logger.info('run {}_{} is finished'.format(date, args.name))
    logger.info('took {}'.format((datetime.datetime.now() - start)))


if __name__ == '__main__':
    main()
