"""
Fit a whole song.

This module fits a whole song!
"""

import argparse as ap
from copy import deepcopy

import numpy as np
from fastdtw import fastdtw
from scipy.io import wavfile

from gesture_fitter import _calc_res, fit_gesture
from synth import gen_sound, only_sin

rng = np.random.RandomState()


def log(*args, **kwargs):
    """Log in stdout, not super smart."""
    print(*args, **kwargs)
    # pass


def logw(*args, **kwargs):
    log('W:', *args, **kwargs)

def logi(*args, **kwargs):
    log('I:', *args, **kwargs)

def logd(*args, **kwargs):
    log('Debug:')
    log(*args, **kwargs)


def default_priors(nb_sin=3):
    """Give the default priors for a gesture fit."""
    prior = []
    for k in range(1, nb_sin + 1):  # prior on sin
            prior.extend([1/k, 1/k, np.pi/(k**2), 10*3**k])
    prior.append(0)
    prior.extend([0, 0, 0, 0, -0.002])  # beta prior
    return np.array(prior)


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


class SongModel:
    """Song model structure."""

    def __init__(self, song, gestures=None, nb_split=20):
        """
        Initialize the song model structure.

        GTE - list of the GTE of the song
        priors - list of the priors of the song for a gesture
        """
        self.song = song
        if gestures is None:
            gestures = [[(i * len(song)) // nb_split, default_priors()]
                        for i in range(nb_split)]
        self.gestures = deepcopy(gestures)

    def mutate(self):
        """Give a new song model with new GTEs."""
        act = rng.uniform()
        gestures = deepcopy(self.gestures)
        if act < 0.02 and len(gestures) > 2:  # Delete a gesture
            logi('deleted')
            to_del = rng.randint(1, len(gestures))
            del gestures[to_del]
        elif act < 0.12:  # Add a new gesture
            logi('added')
            to_add = rng.randint(0, gestures[-1][0])
            gestures.append([to_add, default_priors()])
        elif act < 0.5:  # Take a gesture and put it in another gesture
            logi('copied')
            from_, dest = rng.randint(len(gestures), size=2)
            gestures[dest][1] = deepcopy(gestures[from_][1])
        else:  # Move where the gesture start
            logi('moved')
            to_move = rng.randint(1, len(gestures))
            min_pos = self.gestures[to_move - 1][0] + 100
            try:
                max_pos = self.gestures[to_move + 1][0] - 100
            except IndexError:  # Perhaps we have picked the last gesture
                logd('last gesture picked')
                max_pos = len(self.song) - 100
            new_pos = rng.normal(loc=gestures[to_move][0],
                                 scale=(max_pos-min_pos)/2)
            try:
                logd(self.gestures[to_move - 1][0], '|',
                     self.gestures[to_move][0], '->',
                     new_pos, '|',
                     self.gestures[to_move + 1][0])
            except IndexError:
                pass
            gestures[to_move][0] = int(np.clip(new_pos, min_pos, max_pos))
        # clean GTEs
        gestures.sort(key=lambda x: x[0])
        for i in range(1, len(gestures) - 1):
            # FIXME: Last gesture can be really close to song length
            if gestures[i][0] - gestures[i - 1][0] < 100:
                del gestures[i]
        return SongModel(self.song, gestures)

    def gen_sound(self):
        """Generate the full song."""
        sounds = []
        length = len(self.song)
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
            assert not np.all(np.isnan(sounds[-1])), \
                'only nan in output with p {}'.format(param)
        out = np.concatenate(sounds)
        return out


def fit_song(tutor_song, sr, train_per_day=10, nb_day=5, nb_conc_song=3,
             nb_replay=3, nb_iter_per_train=5, nb_split=10):
    """Fit a song with a day and a night phase."""
    songs = [SongModel(song=tutor_song, nb_split=nb_split)
             for i in range(nb_conc_song)]
    songlog = []

    for iday in range(nb_day):
        logi('☀️️\t☀️️\t☀️️\tDay {} of {}\t☀️️\t☀️️\t☀️'.format(iday+1, nb_day)) #noqa
        songlog.append(('BeforeDay', deepcopy(songs)))
        #######
        # Day #
        #######
        for itrain in range(train_per_day):
            isong = rng.randint(len(songs))
            song = songs[isong]
            ig = rng.randint(len(song.gestures))
            start = song.gestures[ig][0]
            try:
                end = song.gestures[ig + 1][0]
            except IndexError:  # We have picked the last gesture
                end = len(tutor_song)
            logi('{}/{}: fit gesture {} of song {} (length {})'.format(
                itrain+1, train_per_day, ig, isong, end-start))
            prior = deepcopy(song.gestures[ig][1])
            res, score = fit_gesture(
                tutor_song[start:end], start_prior=prior,
                nb_iter=nb_iter_per_train)
            songs[isong].gestures[ig][1] = deepcopy(res)

        #########
        # Night #
        #########
        songlog.append(('BeforeNight', deepcopy(songs)))
        night_songs = deepcopy(songs)
        nb_conc_night = nb_conc_song * 2
        if iday + 1 != nb_day:
            logi('💤\t💤\t💤\tNight\t💤\t💤\t💤')
            score = get_scores(tutor_song, night_songs, sr)
            logi('scores:', score.astype(int))
            logi('ranks:', rank(score))
            fitness = (len(night_songs)) - rank(score)
            night_songs = np.random.choice(night_songs, size=nb_conc_night,
                                           p=fitness/np.sum(fitness))
            for ireplay in range(nb_replay):
                print('mutation {} out of {}'.format(ireplay, nb_replay))
                night_songs = [song.mutate() for song in night_songs]
            songs = rng.choice(night_songs, size=nb_conc_song, replace=False)
            songs = songs.tolist()
    songlog.append(('End', songs))
    return songs, songlog


def get_scores(tutor_song, song_models, sr):
    """
    Get the score of each model compared to the tutor song.

    tutor_song - The signal of the tutor song
    song_models - A list of all the song models
    """
    g = _calc_res(tutor_song, sr)
    scores = np.zeros(len(song_models))

    for i in range(len(song_models)):
        sig = song_models[i].gen_sound()
        c = deepcopy(_calc_res(sig, sr))
        scores[i] = fastdtw(g, c, radius=3, dist=2)[0]
    return scores


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
    try:
        songs, song_log = fit_song(tsong, sr, train_per_day=args.train_per_day,
                                   nb_day=args.days, nb_conc_song=args.concurrent,
                                   nb_iter_per_train=args.iter_per_train,
                                   nb_split=30,
                                   nb_replay=args.replay)
    except KeyboardInterrupt as e:
        f = open(os.path.join(path, 'aborted.txt'), 'a')
        f.write('aborted')
        f.close()
        raise e
    logi('!!!! Learning over !!!!')
    logi('Logging the songs')
    with open(os.path.join(path, 'songs.pkl'), 'wb') as f:
        pickle.dump(songs, f)
    with open(os.path.join(path, 'songlog.pkl'), 'wb') as f:
        pickle.dump(song_log, f)
    logi('Generating the waves')
    for i, song in enumerate(songs):
        wavfile.write(os.path.join(path, 'out_{}.wav'.format(i)),
                      44100, song.gen_sound())
    logi('run {}_{} is finished'.format(date, args.name))
    logi('took {}'.format((datetime.datetime.now() - start)))


if __name__ == '__main__':
    main()
