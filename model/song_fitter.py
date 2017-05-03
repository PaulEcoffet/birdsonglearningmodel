"""Fit a whole song.

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
import sys
from shutil import copyfile
from subprocess import call

import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile
from datasaver import DataSaver, QuietDataSaver

from day_optimisers import optimise_gesture_dummy, optimise_gesture_padded,\
                           optimise_gesture_whole
from measures import bsa_measure, get_scores
from night_optimisers import mutate_best_models_dummy, \
                             mutate_best_models_elite, \
                             mutate_microbial, \
                             mutate_microbial_extended_elite, \
                             mutate_microbial_extended_uniform
from song_model import SongModel


logger = logging.getLogger('root')
EDITOR = os.environ.get('EDITOR', 'vim')

DAY_LEARNING_MODELS = {
    'optimise_gesture_dummy': optimise_gesture_dummy,
    'optimise_gesture_padded': optimise_gesture_padded,
    'optimise_gesture_whole': optimise_gesture_whole
}
NIGHT_LEARNING_MODELS = {
    'mutate_best_models_dummy': mutate_best_models_dummy,
    'mutate_best_models_elite': mutate_best_models_elite,
    'mutate_microbial': mutate_microbial,
    'mutate_microbial_extended_elite': mutate_microbial_extended_elite,
    'mutate_microbial_extended_uniform': mutate_microbial_extended_uniform
}
COMP_METHODS = {'linalg': lambda g, c: np.linalg.norm(g - c),
                'fastdtw': lambda g, c: fastdtw(g, c, dist=2, radius=1)[0]}


def fit_song(tutor_song, conf, datasaver=None):
    # FIXME OUTDATED
    """Fit a song with a day and a night phase.

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
    day_optimisation = DAY_LEARNING_MODELS[conf['dlm']]
    night_optimisation = NIGHT_LEARNING_MODELS[conf['nlm']]
    nb_day = conf['days']
    nb_conc_song = conf['concurrent']
    measure = conf['measure_obj']
    comp = conf['comp_obj']
    rng = conf['rng_obj']
    nb_split = 30

    songs = [SongModel(song=tutor_song, priors=conf['prior'],
                       nb_split=nb_split, rng=rng)
             for i in range(nb_conc_song)]
    if datasaver is None:
        datasaver = QuietDataSaver()
    datasaver.add(moment='Start', songs=songs,
                  scores=get_scores(tutor_song, songs, measure, comp))

    for iday in range(nb_day):
        logger.info('☀️️\t☀️️\t☀️️\tDay {} of {}\t☀️️\t☀️️\t☀️'.format(iday+1, nb_day)) # noqa
        with datasaver.set_context('day_optim'):
            songs = day_optimisation(songs, tutor_song, conf,
                                     datasaver=datasaver)
        score = get_scores(tutor_song, songs, measure, comp)
        if iday + 1 != nb_day:
            logger.debug(score)
            datasaver.add(moment='BeforeNight',
                          songs=songs, scores=score)
            logger.info('💤\t💤\t💤\tNight\t💤\t💤\t💤')
            with datasaver.set_context('night_optim'):
                songs = night_optimisation(songs,
                                           tutor_song, conf,
                                           datasaver=datasaver)
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
    global NIGHT_LEARNING_MODELS, DAY_LEARNING_MODELS, COMP_METHODS
    logging.basicConfig(level=logging.DEBUG)

    start = datetime.datetime.now()
    tsong = None
    parser = ap.ArgumentParser(
        description="""
        reproduce the learning of a zebra finch for a given tutor song.
        """
    )
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
    parser.add_argument('--comp', type=str, required=False, default='linalg',
                        choices=COMP_METHODS,
                        help='comparison method to use')
    parser.add_argument('--dlm', type=str, required=False,
                        choices=DAY_LEARNING_MODELS, help="day learning model")
    parser.add_argument('--nlm', type=str, required=False,
                        choices=NIGHT_LEARNING_MODELS,
                        help="night learning model")
    parser.add_argument('--edit-conf', action='store_true')
    parser.add_argument('--coefs', type=ap.FileType('r'),
                        default='confs/default_coefs.json',
                        help="file with the coefs")
    parser.add_argument('--priors', type=ap.FileType('r'),
                        default="confs/default_prior_max_min_dev.json")
    parser.add_argument('--no-desc', dest='edit_desc', action='store_false')
    args = parser.parse_args()
    if args.seed is None:
        seed = int(datetime.datetime.now().timestamp())
    else:
        seed = args.seed
    rng = np.random.RandomState(seed)
    conf = {}
    if args.config:
        conf.update(json.load(args.config))
        try:  # Warning if reproduction (with commit key) and different commits
            if conf['commit'] != get_git_revision_hash():
                logger.warning('Commit recommended for the conf is different'
                               ' from the current commit.')
        except KeyError:
            pass
        try:
            sr, tsong = wavfile.read(conf['tutor'])
        except KeyError:
            pass
        conf['commit'] = get_git_revision_hash()
    argdata = {'days': args.days,
               'train_per_day': args.train_per_day,
               'concurrent': args.concurrent,
               'name': args.name,
               'seed': seed,
               'replay': args.replay,
               'iter_per_train': args.iter_per_train,
               'commit': get_git_revision_hash(),
               'comp': args.comp,
               'dlm': args.dlm,
               'nlm': args.nlm}
    if args.tutor is not None:
        argdata['tutor'] = args.tutor.name
    if tsong is None:
        sr, tsong = wavfile.read(args.tutor)

    conf.update({k: v for k, v in argdata.items() if v is not None})

    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    run_name = '{}_{}'.format(conf['name'], date)
    path = 'res/{}'.format(run_name)
    os.makedirs(path)
    wavfile.write(os.path.join(path, 'tutor.wav'), sr, tsong)
    pmmd = json.load(args.priors)
    conf.update(pmmd)
    coefs = json.load(args.coefs)
    conf.update(coefs)
    if args.edit_desc:
        write_run_description(path)
    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump({k: conf[k] for k in conf if not k.endswith('_obj')},
                  f, indent=4)  # human readable parameters
    if args.edit_conf:
        call([EDITOR, os.path.join(path, 'params.json')])
        with open(os.path.join(path, 'params.json'), 'r') as f:
            conf = json.load(f)

    datasaver = DataSaver(defaultdest=os.path.join(path, 'data_cur.pkl'))
    logger.info(pformat(conf))

    conf['rng_obj'] = rng
    conf['measure_obj'] = lambda x: bsa_measure(x, 44100)
    conf['comp_obj'] = COMP_METHODS[conf['comp']]

    #########################################
    # STOP READING CONF; START THE LEARNING #
    #########################################
    try:
        songs = fit_song(tsong, conf, datasaver=datasaver)
    except KeyboardInterrupt as e:
        logger.warning('Aborted')
        with open(os.path.join(path, 'aborted.txt'), 'a') as f:
            f.write('aborted\n')
    finally:
        logger.info('Saving the data.')
        datasaver.write(os.path.join(path, 'data.pkl'))
    logger.info('!!!! Learning over !!!!')
    try:
        subprocess.Popen(['notify-send',
                          '{} is finished'.format(run_name)])
    except OSError:
        pass
    total_time = datetime.datetime.now() - start
    logger.info('Run {} is over. Took {}'.format(run_name, total_time))


def write_run_description(path):
    """Open an editor with a prefilled file to describe the run."""
    copyfile('desc.template.md', os.path.join(path, 'desc.md'))
    call([EDITOR, os.path.join(path, 'desc.md')])


if __name__ == '__main__':
    main()
