"""
Fit a whole song.

This module fits a whole song!
"""

import argparse as ap
import logging
import os
import datetime
import pickle
import json
import subprocess
from pprint import pformat

import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile
from datasaver import DataSaver, QuietDataSaver

from day_optimisers import optimise_gesture_dummy
from measures import bsa_measure, get_scores
from night_optimisers import mutate_best_models_dummy, mutate_best_models_elite
from song_model import SongModel


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('root')


def fit_song(tutor_song, measure, comp, day_optimisation, night_optimisation,
             day_conf, night_conf, nb_day=5, nb_conc_song=3, nb_split=10,
             datasaver=None, rng=None):
    """
    Fit a song with a day and a night phase.

    tutor_song - Format: 1D np.array
                 The tutor song that the algorithm will try to reproduce.
                 It will be normalized between -1 and +1 internally.
                 You don't need to do it yourself.
    measure - Format: a function taking a song or a gesture as argument.
              signature : measure(1D-np.array) -> `comp` argument.
              The function to measure the features of the tutor songs
              and the generated songs. The module `measures` contains
              several measures that can be used. The classical ones are
              `lambda x: birdsonganalysis.all_song_features(x, samplerate)` and
              `lambda x: python_speech_features.mfcc(x, ...)`.
    comp - Format: a function taking the tutor song measure and a generated
           song measure and return a real score.
           Signature : comp(goal_measure, cur_measure) -> float
           `lambda g, c: np.linalg.norm(g - c)` and
           `lambda g, c: fastdtw.fastdtw(g, c)[0]` are common comparison
           function.
    day_optimisation - Format: function taking a list of SongModel and returns
                       a list of SongModel.
                       Signature: day_optimisation(list(SongModel), **day_conf)
                                        ->list(SongModel)
                       This function does the optimisations that are supposed
                       to occur during the day.
    """
    songs = [SongModel(song=tutor_song, nb_split=nb_split, rng=rng)
             for i in range(nb_conc_song)]
    if datasaver is None:
        datasaver = QuietDataSaver()
    datasaver.add(moment='Start', songs=songs,
                  scores=get_scores(tutor_song, songs, measure, comp))

    for iday in range(nb_day):
        logger.info('☀️️\t☀️️\t☀️️\tDay {} of {}\t☀️️\t☀️️\t☀️'.format(iday+1, nb_day)) # noqa
        with datasaver.set_context('day_optim'):
            songs = day_optimisation(songs,
                                     tutor_song, measure, comp,
                                     datasaver=datasaver, rng=rng, **day_conf)
        score = get_scores(tutor_song, songs, measure, comp)
        if iday + 1 != nb_day:
            logger.debug(score)
            datasaver.add(moment='BeforeNight',
                          songs=songs, scores=score)
            logger.info('💤\t💤\t💤\tNight\t💤\t💤\t💤')
            with datasaver.set_context('night_optim'):
                songs = night_optimisation(songs,
                                           tutor_song, measure, comp,
                                           datasaver=datasaver, rng=rng,
                                           **night_conf)
            score = get_scores(tutor_song, songs, measure, comp)
            datasaver.add(moment='AfterNight', songs=songs, scores=score)
        datasaver.write()
    datasaver.add(moment='End', songs=songs,
                  scores=get_scores(tutor_song, songs, measure, comp))
    return songs


def get_git_revision_hash():
    """
    Get the git commit/revision hash.

    Knowing the git revision hash is helpful to reproduce a result with
    the code corresponding to a specific run.
    """
    try:
        return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']),
                   'utf8').strip()
    except OSError:
        return None


def main():
    """Main function for this module, called if not imported."""
    start = datetime.datetime.now()
    tsong = None
    parser = ap.ArgumentParser(
        description="""
        reproduce the learning of a zebra finch for a given tutor song.
        """
    )
    comp_methods = {'linalg': lambda g, c: np.linalg.norm(g - c),
                    'fastdtw': lambda g, c: fastdtw(g, c, radius=10)[0]}
    parser.add_argument('tutor', type=ap.FileType('rb'), nargs='?',
                        help='The targeted song to learn')
    parser.add_argument('--config', type=ap.FileType('r'), required=False,
                        help='The config file to take the parameters from.')
    parser.add_argument('-d', '--days', type=int, required=None,
                        help='number of days')
    parser.add_argument('-t', '--train-per-day', type=int, required=False,
                        help='number of training for a gesture per day')
    parser.add_argument('-c', '--concurrent', type=int, required=False,
                        help='number of concurrent model for the song')
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='name of the trial for logging')
    parser.add_argument('-s', '--seed', type=int, required=False,
                        help='seed for the random number generator')
    parser.add_argument('-r', '--replay', type=int, required=False,
                        help='number of passes for new generations during'
                        ' night')
    parser.add_argument('-i', '--iter-per-train', type=int, required=False,
                        help='number of iteration when training a gesture')
    parser.add_argument('--comp', type=str, required=False,
                        choices=comp_methods,
                        help='comparison method to use')
    args = parser.parse_args()
    if args.seed is None:
        seed = int(datetime.datetime.now().timestamp())
    else:
        seed = args.seed
    rng = np.random.RandomState(seed)

    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    data = {}
    if args.config:
        data.update(json.load(args.config))
        try:  # Warning if reproduction (with commit key) and different commits
            if data['commit'] != get_git_revision_hash():
                logger.warning('Commit recommended for the data is different'
                               ' from the current commit.')
        except KeyError:
            pass
        try:
            sr, tsong = wavfile.read(data['tutor'])
        except KeyError:
            pass
        data['commit'] = get_git_revision_hash()
    argdata = {'days': args.days,
               'train_per_day': args.train_per_day,
               'concurrent': args.concurrent,
               'name': args.name,
               'seed': seed,
               'replay': args.replay,
               'iter_per_train': args.iter_per_train,
               'commit': get_git_revision_hash(),
               'comp': args.comp}
    if args.tutor is not None:
        argdata['tutor'] = args.tutor.name
    data.update({k: v for k, v in argdata.items() if v is not None})
    if tsong is None:
        sr, tsong = wavfile.read(args.tutor)

    path = 'res/{}_{}'.format(date, data['name'])
    os.makedirs(path)
    wavfile.write(os.path.join(path, 'tutor.wav'), sr, tsong)
    logger.info(pformat(data))
    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump(data, f, indent=4)  # human readable parameters
    day_conf = {'nb_iter_per_train': data['iter_per_train'],
                'train_per_day': data['train_per_day']}
    night_conf = {'nb_replay': data['replay']}
    datasaver = DataSaver(defaultdest=os.path.join(path, 'data_cur.pkl'))
    try:
        songs = fit_song(
            tsong,
            measure=lambda x: bsa_measure(x, sr),
            comp=comp_methods[data['comp']],
            day_optimisation=optimise_gesture_dummy,
            night_optimisation=mutate_best_models_elite,
            day_conf=day_conf,
            night_conf=night_conf,
            nb_day=data['days'],
            nb_conc_song=data['concurrent'],
            nb_split=30,
            datasaver=datasaver,
            rng=rng)
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
    logger.info('run {}_{} is finished'.format(date, data['name']))
    logger.info('took {}'.format((datetime.datetime.now() - start)))
    try:
        subprocess.Popen(['notify-send',
                          '{}_{} is finished'.format(date, data['name'])])
    except OSError:
        pass


if __name__ == '__main__':
    main()
