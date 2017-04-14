"""Analyse syllables."""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import birdsonganalysis as bsa
from copy import deepcopy, copy
import pandas as pd


sns.set_palette('colorblind')


def _running_mean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N


def extract_syllables_feature(song, threshold=None, normalize=True):
    """Extract the features of all syllables and segment them.

    If threshold is None, the threshold will be the one with the biggest jump
    between two amplitude, smoothen by a runnig mean.
    The guess is not always perfect though.
    """
    sfeat = bsa.normalize_features(
        bsa.all_song_features(song, 44100,
                              freq_range=256, fft_size=1024, fft_step=40))
    if threshold is None:
        sort_amp = np.sort(sfeat['amplitude'])
        i_max_diff = np.argmax(_running_mean(np.diff(sort_amp), 100))
        threshold = sort_amp[i_max_diff]
    syllables = []
    beg = None
    end = None
    for i, amp in enumerate(sfeat['amplitude']):
        if beg is None and amp > threshold:
            beg = i
        elif beg is not None and amp < threshold:
            end = i
            if end - beg > 15:
                syllable_dict = {'beg': beg, 'end': end}
                for key in sfeat:
                    syllable_dict[key] = deepcopy(sfeat[key][beg:end])
                syllables.append(syllable_dict)
            beg = None
            end = None
    return syllables

def percentage_change(first, last):
    """Compute the percentage of change between two datasets."""
    first = copy(first)
    first['cond'] = 'fist'
    last = copy(last)
    last['cond'] = 'last'
    return ((last.median() - first.median()) / first.median() * 100).abs()


def all_syllables_features(rd: pd.DataFrame):
    """Get all the information from each syllables from all the songs."""
    syllables = []
    for i, row in rd.iterrows():
        moment = row['moment']
        if moment == 'Start' or moment == 'AfterNight':
            moment = 'morning'
        elif moment == 'End' or moment == 'BeforeNight':
            moment = 'evening'
        out = []
        for sm in row['songs']:
            out += extract_syllables_feature(sm.gen_sound())
            for syllable in out:
                syb_dict = {'day': i//2,
                            'moment': moment,
                            'beg': syllable['beg'],
                            'end': syllable['end'],
                            'length': syllable['end'] - syllable['beg']}
                for key in syllable:
                    if key == 'beg' or key == 'end':
                        continue
                    syb_dict['m'+key] = np.mean(syllable[key])
                    syb_dict['v'+key] = np.var(syllable[key])
                syllables.append(syb_dict)
    return pd.DataFrame(syllables)
