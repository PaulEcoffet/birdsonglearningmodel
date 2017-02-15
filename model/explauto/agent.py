import numpy as np
from explauto import Agent
from explauto.evaluation import Evaluation


class BirdAgent(Agent):

    def __init__(self, conf, sm_model, im_model, n_bootstrap=0,
                 context_mode=None, seed=None):
        super().__init__(conf, sm_model, im_model, n_bootstrap=0,
                       context_mode=None)
        self.gte = None
        self.mfcc_song = None
        self.rng = np.random.RandomState(seed)

    def define_goals(self, tutor_song, itermax, env):
        if self.gte is None:
            self.gte = self.rng.randint(0, len(tutor_song), size=30)
            self.gte = self.clean_gte(self.gte)
        best_gte = self.gte
        best_score = self.score_gte(best_gte, tutor_song, env)
        print('previous score:', best_score)
        for cur_iter in range(itermax):
            cur_gte = np.copy(best_gte)
            action = self.rng.uniform()
            if action < 0.1:  # add GTE
                i = self.rng.randint(len(cur_gte)-1)
                cur_gte = np.append(cur_gte, (cur_gte[i]+cur_gte[i+1])//2)
                cur_gte = self.clean_gte(cur_gte)
            elif action < 0.2:  # remove GTE
                i = self.rng.randint(len(cur_gte))
                cur_gte = np.delete(cur_gte, i)
            else:  # move GTE
                i = self.rng.randint(len(cur_gte))
                cur_gte[i] += self.rng.randint(-30, 30)
                np.clip(cur_gte, 0, len(tutor_song), out=cur_gte)
                cur_gte = self.clean_gte(cur_gte)
            cur_score = self.score_gte(cur_gte, tutor_song, env)
            if cur_score < best_score:
                best_gte = np.copy(cur_gte)
                best_score = cur_score
        if np.any(best_gte != self.gte):
            print('new cut better than previous one')
            print('new score:', best_score)
            self.gte = best_gte
            self.mfcc_song = self.get_mfcc_from_gte(self.gte, tutor_song, env)

    def score_gte(self, gte, tutor_song, env):
        tests = self.get_mfcc_from_gte(gte, tutor_song, env)
        return np.sum(Evaluation(self, env, tests).evaluate())

    def get_mfcc_from_gte(self, gte_, tutor_song, env):
        gte = np.copy(gte_)
        if gte[0] != 0:
            gte = np.insert(gte, 0, 0)
        if gte[-1] != len(tutor_song):
            gte = np.append(gte, len(tutor_song))
        out = np.zeros((len(gte), self.conf.s_ndims))
        for i in range(len(gte) - 1):
            out[i, :] = env.listen(tutor_song[gte[i]:gte[i+1]])
        return out

    def clean_gte(self, gte):
        gte = np.sort(gte)
        clean = False
        while not clean:
            clean = True
            i = 0
            removed = []
            while i < len(gte) - 1:
                if gte[i+1] - gte[i] < 500:
                    clean = False

                    removed.append(i + 1)
                    i += 1
                i += 1
            gte = np.delete(gte, removed)
        return gte

    def choose(self):
        if self.mfcc_song is not None:
            return self.mfcc_song[self.rng.randint(len(self.mfcc_song)), :]
        else:
            return super().choose()
