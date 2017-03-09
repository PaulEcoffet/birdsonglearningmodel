"""
Fit a whole song.

This module fits a whole song!
"""

import argparse as ap
import logging
from copy import deepcopy
import os
import datetime
import pickle
import json
import subprocess

import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile
from datasaver import DataSaver, QuietDataSaver

from day_optimisers import optimise_gesture_dummy
from measures import bsa_measure, get_scores
from night_optimisers import mutate_best_models_dummy
from song_model import SongModel

rng = np.random.RandomState()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('root')


def fit_song(tutor_song, measure, comp, day_optimisation, night_optimisation,
             day_conf, night_conf, nb_day=5, nb_conc_song=3, nb_split=10,
             datasaver=None):
    """Fit a song with a day and a night phase."""
    songs = [SongModel(song=tutor_song, nb_split=nb_split, rng=rng)
             for i in range(nb_conc_song)]
    if datasaver is None:
        datasaver = QuietDataSaver()
    datasaver.add(moment='Start', songs=songs,
                  scores=get_scores(tutor_song, songs, measure, comp))

    for iday in range(nb_day):
        logger.info('☀️️\t☀️️\t☀️️\tDay {} of {}\t☀️️\t☀️️\t☀️'.format(iday+1, nb_day)) # noqa
        with datasaver.set_context('day_optim'):
            songs = day_optimisation(deepcopy(songs),
                                     tutor_song, measure, comp,
                                     datasaver=datasaver, **day_conf)
        score = get_scores(tutor_song, songs, measure, comp)
        if iday + 1 != nb_day:
            logger.debug(score)
            datasaver.add(moment='BeforeNight',
                          songs=deepcopy(songs), scores=deepcopy(score))
            logger.info('💤\t💤\t💤\tNight\t💤\t💤\t💤')
            with datasaver.set_context('night_optim'):
                songs = night_optimisation(deepcopy(songs),
                                           tutor_song, measure, comp,
                                           datasaver=datasaver, **night_conf)
            score = get_scores(tutor_song, songs, measure, comp)
            datasaver.add(moment='AfterNight', songs=songs, scores=score)
    datasaver.add(moment='End', songs=songs,
                  scores=get_scores(tutor_song, songs, measure, comp))
    return songs


def get_git_revision_hash():
    try:
        return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']),
                   'utf8').strip()
    except OSError:
        return None

def main():
    """Main function for this module, called if not imported."""
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
    data = {'days': args.days,
            'train_per_day': args.train_per_day,
            'concurrent': args.concurrent,
            'name': args.name,
            'seed': seed,
            'replay': args.replay,
            'iter_per_train': args.iter_per_train,
            'commit': get_git_revision_hash()}
    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump(data, f, indent=4)  # human readable parameters
    day_conf = {'nb_iter_per_train': args.iter_per_train,
                'train_per_day': args.train_per_day}
    night_conf = {'nb_replay': args.replay}
    datasaver = DataSaver()
    try:
        songs = fit_song(
            tsong,
            measure=lambda x: bsa_measure(x, sr),
            comp=lambda g, c: fastdtw(g - c, radius=10),
            day_optimisation=optimise_gesture_dummy,
            night_optimisation=mutate_best_models_dummy,
            day_conf=day_conf,
            night_conf=night_conf,
            nb_day=args.days,
            nb_conc_song=args.concurrent,
            nb_split=30,
            datasaver=datasaver)
    except KeyboardInterrupt as e:
        logger.warning('Aborted')
        with open(os.path.join(path, 'aborted.txt'), 'a') as f:
            f.write('aborted')
    finally:
        logger.info('Saving the data.')
        datasaver.write(os.path.join(path, 'data.pkl'))
    logger.info('!!!! Learning over !!!!')
    logger.info('Logging the songs')
    with open(os.path.join(path, 'songs.pkl'), 'wb') as f:
        pickle.dump(songs, f)
    logger.info('Generating the waves')
    for i, song in enumerate(songs):
        wavfile.write(os.path.join(path, 'out_{}.wav'.format(i)),
                      44100, song.gen_sound())
    logger.info('run {}_{} is finished'.format(date, args.name))
    logger.info('took {}'.format((datetime.datetime.now() - start)))


if __name__ == '__main__':
    main()
