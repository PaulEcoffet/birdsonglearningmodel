import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_palette('colorblind')

def draw_learning_curve(log, ax=None):
    root_data = [item[1] for item in log if item[0] == 'root']
    rd = pd.DataFrame(root_data)
    score_array = np.array([list(a) for a in rd['scores']]).T
    if ax is None:
        fig = plt.figure(figsize=(16, 5))
        ax = fig.gca()
    for i in range(1, len(rd['scores']), 2):
        ax.axvspan(i, i+1, facecolor='darkblue', alpha=0.1)
    sns.tsplot(score_array, err_style='unit_traces', ax=ax)
    return ax
