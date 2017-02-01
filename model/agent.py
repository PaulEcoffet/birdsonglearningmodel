import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
from gte import extract_gte

from explauto import SensorimotorModel, Agent, InterestModel
from explauto import Experiment

from environment import BirdSongEnvironment

environment = BirdSongEnvironment()
sm_model = SensorimotorModel.from_configuration(environment.conf,
                                                'nearest_neighbor', 'default')
im_model = InterestModel.from_configuration(environment.conf,
                                            environment.conf.s_dims,
                                            'random')
agent = Agent(environment.conf, sm_model, im_model)
expe = Experiment(environment, agent)


def testcase():
    fname = '../data/ba_example.wav'
    sr, signal = wavfile.read(fname)
    gtes = np.concatenate((np.array([0]),
                           extract_gte(fname),
                           np.array([len(signal)-1])))
    tests = np.zeros((len(gtes), len(environment.conf.s_maxs)))
    i = 0
    c = 0
    while i < len(gtes) - 1:
        j = i + 1
        while j < len(gtes)-1 and gtes[j] - gtes[i] < 500:
            j += 1
        tests[c, :] = \
            environment.listen(signal[gtes[i]:gtes[j]])
        c += 1
        i = j
    return tests[:c, :]

expe.evaluate_at([20, 40, 80, 200, 400], testcase())

expe.run()

fig = plt.figure()
ax = fig.gca()
expe.log.plot_learning_curve(ax)
plt.show(fig)
