import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from explauto import SensorimotorModel, Agent, InterestModel
from explauto import Experiment

from environment import BirdSongEnvironment

def testcase(configstr='', nb_samples=100):
    tests = np.random.randint(0, 10,
                              size=(nb_samples, len(environment.conf.s_maxs)))
    tests[:, 0] = 0.5
    return tests


environment = BirdSongEnvironment()

sm_model = SensorimotorModel.from_configuration(environment.conf,
                                                'nearest_neighbor', 'default')
im_model = InterestModel.from_configuration(environment.conf,
                                            environment.conf.s_dims,
                                            'random')

agent = Agent(environment.conf, sm_model, im_model)

expe = Experiment(environment, agent)

expe.evaluate_at([20, 40, 80, 200, 400], testcase())

expe.run()

fig = plt.figure()
ax = fig.gca()
expe.log.plot_learning_curve(ax)
plt.show(fig)
