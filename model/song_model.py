"""Define the SongModel class."""

from copy import deepcopy

import numpy as np
import logging
from synth import only_sin, gen_alphabeta, synthesize
from gesture_fitter import _padded_gen_sound


logger = logging.getLogger('songmodel')


def default_priors(nb_sin=3):
    """Give the default priors for a gesture fit."""
    prior = []
    for k in range(1, nb_sin + 1):  # prior on sin
        prior.extend([0, 0, np.pi/(k**2), 3**k])
    prior.append(0)
    prior.extend([0, 0, 0, 0, 0.15])  # beta prior
    return np.array(prior)


class SongModel:
    """Song model structure."""

    def __init__(self, song, gestures=None, nb_split=20, rng=None,
                 parent=None):
        """
        Initialize the song model structure.

        GTE - list of the GTE of the song
        priors - list of the priors of the song for a gesture
        """
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        self.song = song
        if gestures is None:
            gestures = [[(i * len(song)) // nb_split, default_priors()]
                        for i in range(nb_split)]
        self.gestures = deepcopy(gestures)
        # Do not keep track of parent for now, avoid blow up in copy
        self.parent = None

    def mutate(self, n=1):
        """Give a new song model with new GTEs."""
        gestures = deepcopy(self.gestures)
        for i in range(n):
            act = self.rng.uniform()
            if act < 0.02 and len(gestures) > 2:  # Delete a gesture
                logger.info('deleted')
                to_del = self.rng.randint(1, len(gestures))
                del gestures[to_del]
            elif act < 0.3:  # Add a new gesture
                logger.info('added')
                to_add = self.rng.randint(0, gestures[-1][0] - 100)
                gestures.append([to_add, default_priors()])
            elif act < 0.5:  # Take a gesture and put it in another gesture
                logger.info('copied')
                from_, dest = self.rng.randint(len(gestures), size=2)
                gestures[dest][1] = deepcopy(gestures[from_][1])
            elif act < 0.8:  # Move where the gesture start
                logger.info('moved')
                to_move = self.rng.randint(1, len(gestures))
                min_pos = gestures[to_move - 1][0] + 100
                try:
                    max_pos = gestures[to_move + 1][0] - 100
                except IndexError:  # Perhaps we have picked the last gesture
                    logger.debug('last gesture picked')
                    max_pos = len(self.song) - 100
                new_pos = self.rng.normal(loc=gestures[to_move][0],
                                          scale=(max_pos-min_pos)/4)
                gestures[to_move][0] = int(np.clip(new_pos, min_pos, max_pos))
            else:  # Do not mutate
                pass
            # clean GTEs
            gestures.sort(key=lambda x: x[0])
            clean = False
            while not clean:
                for i in range(1, len(gestures)):
                    if gestures[i][0] - gestures[i - 1][0] < 100:
                        del gestures[i]
                        break
                else:  # If there is no break (for/else python syntax)
                    clean = True
            if len(self.song) - gestures[-1][0] < 100:
                del gestures[-1]
        return SongModel(self.song, gestures, parent=self)

    def gen_sound(self, range_=None):
        """Generate the full song."""
        ab = self.gen_alphabeta(range_=range_, pad='last')
        out = synthesize(ab)
        assert np.isclose(np.nanmean(out), 0)
        if range_ is not None:
            expected_len = self.gesture_end(range_[-1]) - self.gestures[range_[0]][0]
        else:
            expected_len = len(self.song)
        assert len(out) == expected_len
        if range_ is not None:
            cmp = _padded_gen_sound(self, range_, range_.start, self.gestures[range_.start][1])
        return out

    def gen_alphabeta(self, range_=None, pad=False):
        """Compute alpha and beta for the whole song."""
        if range_ is None:
            range_ = range(len(self.gestures))
        inner_pad = False
        length = self.gesture_end(range_[-1]) - self.gestures[range_[0]][0]
        if pad == 'last':
            length += 2
        elif pad:
            length += 2 * len(range_)
            inner_pad = True
        ab = np.zeros((length, 2))
        # true_start = When the first gesture starts
        true_start = self.gestures[range_[0]][0]
        for i in range_[:-1]:
            params = self.gestures[i][1]
            start = self.gestures[i][0] - true_start  # correct padding
            end = self.gesture_end(i) - true_start
            size = end - start
            if pad is True:
                end += 2
            assert size != 0
            ab[start:end] = gen_alphabeta(
                params, size,
                falpha=lambda x, p: only_sin(x, p, nb_sin=3),
                fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
                falpha_nb_args=13, pad=inner_pad, beg=self.gestures[i][0])
        i = range_[-1]
        params = self.gestures[i][1]
        start = self.gestures[i][0] - true_start  # correct padding
        end = self.gesture_end(i) - true_start
        size = end - start
        if pad == 'last' or pad is True:
            end += 2
        ab[start:end] = gen_alphabeta(
            params, size,
            falpha=lambda x, p: only_sin(x, p, nb_sin=3),
            fbeta=lambda x, p: only_sin(x, p, nb_sin=1),
            falpha_nb_args=13, pad=pad, beg=self.gestures[i][0])
        assert np.all(ab[:, 0] >= 0)
        return ab

    def gesture_end(self, i):
        """Return the end of a gesture."""
        if i < 0:
            i = len(self.gestures) - i
        try:
            end = self.gestures[i + 1][0]
        except IndexError:
            end = len(self.song)
        return end
