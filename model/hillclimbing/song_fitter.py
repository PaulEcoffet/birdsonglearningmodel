"""
Fit a whole song.

This module fits a whole song!
"""

import numpy as np
from copy import deepcopy
from fastdtw import fastdtw
import argparse as ap
from scipy.io import wavfile

from gesture_fitter import fit_gesture, _calc_res
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
            prior.extend([1/k, 3/k, np.pi/(k**2), 10*3**k])
    prior.append(4)
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

    def __init__(self, gestures=None, song=None, nb_split=20):
        """
        Initialize the song model structure.

        GTE - list of the GTE of the song
        priors - list of the priors of the song for a gesture
        """
        if gestures is None:
            gestures = [[(i * len(song)) // nb_split, default_priors()]
                        for i in range(nb_split)]
        self.gestures = deepcopy(gestures)

    def mutate(self):
        """Give a new song model with new GTEs."""
        act = rng.uniform()
        gestures = deepcopy(self.gestures)
        if act < 0.1 and len(gestures) > 2:  # Delete a gesture
            to_del = rng.randint(1, len(gestures))
            del gestures[to_del]
        elif act < 0.2:  # Add a new gesture
            to_add = rng.randint(0, gestures[-1][0])
            gestures.append([to_add, default_priors()])
        elif act < 0.3:  # Take a gesture and put it in another gesture
            from_, dest = rng.randint(1, len(gestures), size=2)
            gestures[dest][1] = deepcopy(gestures[from_][1])
        else:  # Move where the gesture start
            # FIXME: Can go over the song length
            to_move = rng.randint(1, len(gestures))
            gestures[to_move][0] += rng.randint(-500, 500)
        # clean GTEs
        gestures.sort(key=lambda x: x[0])
        for i in range(1, len(gestures) - 1):
            # FIXME: Last gesture can be really close to song length
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
            assert not np.all(np.isnan(sounds[-1])), \
                'only nan in output with p {}'.format(param)
        out = np.concatenate(sounds)
        return out


def fit_song(tutor_song, sr, train_per_day=10, nb_day=5, nb_conc_song=3,
             nb_replay=3, nb_iter_per_train=5):
    """Fit a song with a day and a night phase."""
    songs = [SongModel(song=tutor_song) for i in range(nb_conc_song)]

    for iday in range(nb_day):
        logi('☀️️\t☀️️\t☀️️\tDay {} of {}\t☀️️\t☀️️\t☀️'.format(iday+1, nb_day)) #noqa
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
            assert np.any(res != songs[isong].gestures[ig][1])
            songs[isong].gestures[ig][1] = deepcopy(res)
            assert np.all(songs[isong].gestures[ig][1] == res)

        #########
        # Night #
        #########
        night_songs = deepcopy(songs)
        nb_conc_night = nb_conc_song * 2
        if iday + 1 != nb_day:
            logi('💤\t💤\t💤\tNight\t💤\t💤\t💤')
            for ireplay in range(nb_replay):
                logi('Replay {} out of {}'.format(ireplay + 1, nb_replay))
                score = get_scores(tutor_song, night_songs)
                logi('scores:', score)
                logi('ranks:', rank(score))
                fitness = (len(night_songs)) - rank(score)
                nsongs = np.random.choice(night_songs, size=nb_conc_night,
                                          p=fitness/np.sum(fitness))
                night_songs = [song.mutate() for song in nsongs]
            score = get_scores(tutor_song, night_songs)
            songs = []
            for i in np.argsort(score)[:nb_conc_song]:
                songs.append(night_songs[i])
    return songs


def get_scores(tutor_song, song_models):
    """
    Get the score of each model compared to the tutor song.

    tutor_song - The signal of the tutor song
    song_models - A list of all the song models
    """
    g = _calc_res(tutor_song, sr)
    scores = np.zeros(len(song_models))

    for i in range(len(song_models)):
        sig = song_models[i].gen_sound(len(tutor_song))
        c = deepcopy(_calc_res(sig, sr))
        scores[i] = fastdtw(g, c, radius=3, dist=2)[0]
    return scores


if __name__ == '__main__':
    import os
    import datetime
    import pickle
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
    date = datetime.datetime.now().strftime('%y%m%d_%H%M%s')
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
    songs = fit_song(tsong, sr, train_per_day=args.train_per_day,
                     nb_day=args.days, nb_conc_song=args.concurrent,
                     nb_iter_per_train=args.iter_per_train)
    logi('!!!! Learning over !!!!')
    logi('Logging the songs')
    with open(os.path.join(path, 'songs.pkl'), 'wb') as f:
        pickle.dump(songs, f)
    logi('Generating the waves')
    for i, song in enumerate(songs):
        wavfile.write(os.path.join(path, 'out_{}.wav'.format(i)),
                      44100, song.gen_sound(len(tsong)))
