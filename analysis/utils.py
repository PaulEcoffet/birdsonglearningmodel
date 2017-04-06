import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import pandas as pd
from collections import defaultdict
import io
import base64
from IPython.display import Audio
from ipywidgets import widgets
import copy
import json
import pickle
import sys
from scipy.io import wavfile
import birdsonganalysis as bsa

sys.path.append('../model')

from song_model import SongModel

sns.set_palette('colorblind')


def draw_learning_curve(rd, ax=None):
    score_array = np.array([list(a) for a in rd['scores']]).T
    if ax is None:
        fig = plt.figure(figsize=(16, 5))
        ax = fig.gca()
    for i in range(1, len(rd['scores']), 2):
        ax.axvspan(i, i+1, facecolor='darkblue', alpha=0.1)
    sns.tsplot(score_array, err_style='unit_traces', ax=ax)
    return ax


def plot_to_html(fig):
    # write image data to a string buffer and get the PNG image bytes
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return widgets.HTML("""<img src='data:image/png;base64,{}'/>""".format(
        base64.b64encode(buf.getvalue()).decode('ascii')))


class NoDataException(Exception): pass


class CacheDict(defaultdict):

    def __missing__(self, key):
        return self.default_factory(key)


class GridAnalyser:
    """Analyser for the grid search."""

    def __init__(self, run_paths):
        self.data = CacheDict(lambda i: self._get_data(i))
        self.conf = CacheDict(lambda i: self._get_conf(i))
        self.rd = CacheDict(lambda i: self._get_rd(i))
        self.run_paths = run_paths

    def show(self, i, vbox):
        try:
            best = np.argmax(self.rd[i]['scores'].iloc[-1])
            vbox.children = [
                self.title(i),
                self.audio(i, -1, best),
                self.configuration(i),
                self.learning_curve(i),
                self.spec_deriv_plot(i, -1, best),
                self.tutor_spec_plot()
            ]
        except NoDataException:
            vbox.children = [
                widgets.HTML('<p>No data yet</p>')
            ]

    def audio(self, irun, iday, ismodel):
        a = Audio(self.rd[irun]['songs'].iloc[iday][ismodel].gen_sound(),
                  rate=44100)
        return widgets.HTML(a._repr_html_())

    def spec_deriv_plot(self, irun, iday, ismodel):
        song = self.rd[irun]['songs'].iloc[iday][ismodel].gen_sound()
        fig = plt.figure(figsize=(13, 4))
        ax = fig.gca()
        ax = bsa.spectral_derivs_plot(bsa.spectral_derivs(song, 256, 40, 1024),
                                      contrast=0.01, ax=ax)
        plt.close(fig)
        return plot_to_html(fig)

    def tutor_spec_plot(self):
        sr, tutor = wavfile.read(join(self.run_paths[0], 'tutor.wav'))
        fig = plt.figure(figsize=(13, 4))
        ax = fig.gca()
        bsa.spectral_derivs_plot(bsa.spectral_derivs(tutor, 256, 40, 1024),
                                 contrast=0.01, ax=ax)
        plt.close(fig)
        return plot_to_html(fig)


    def title(self, i):
        return widgets.HTML('<h3>' + self.conf[i]['name'] + '</h3>')

    def configuration(self, i):
        table = widgets.HTML('<table>')
        table.value += '<tr><th>Key</th><th>Value</th></tr>'
        for key in self.conf[i]:
            table.value += '<tr><td>{}</td><td>{}</td>'.format(key, self.conf[i][key])
        table.value += '</table>'
        return table


    def learning_curve(self, i):
        fig = plt.figure(figsize=(13, 4))
        ax = fig.gca()
        try:
            ax = draw_learning_curve(self.rd[i], ax)
        finally:
            plt.close(fig)
        return plot_to_html(fig)

    def _get_data(self, i):
        try:
            with open(join(self.run_paths[i], 'data.pkl'), 'rb') as f:
                out = pickle.load(f)
        except FileNotFoundError:
            try:
                with open(join(self.run_paths[i], 'data_cur.pkl'), 'rb') as f:
                    out = pickle.load(f)
            except FileNotFoundError:
                raise NoDataException
        return out

    def _get_rd(self, i):
        root_data = [item[1] for item in self.data[i] if item[0] == 'root']
        return pd.DataFrame(root_data)

    def _get_conf(self, i):
        with open(join(self.run_paths[i], 'conf.json')) as f:
            conf = json.load(f)
        return conf
