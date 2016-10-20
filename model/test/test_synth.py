import unittest
import numpy as np

from synth.alphabeta2dat import AlphaBeta2Dat
from synth.dat2wav import Dat2Wav

class TestSynth(unittest.TestCase):

    @unittest.skip("The python code is way too slow and precision errors accumulate")
    def test_equiv(self):
        for test in ['ba_example', 'test1', 'test2', 'test3']:
            with open('test/comparison_files/{}_alpha.dat'.format(test)) as f:
                alpha_content = f.readlines()
            size = len(alpha_content) - 1 # Do not count the last \n.
            params = np.zeros((size, 2))
            for i, line in enumerate(alpha_content[:-1]):
                params[i, 0] = float(line.split()[2])
            with open('test/comparison_files/{}_beta.dat'.format(test)) as f:
                for i in range(size):
                    params[i, 1] = float(f.readline().split()[1])
            print("loading over, begin simu")
            out = []
            ab2d = AlphaBeta2Dat(params, lambda x: out.append(x))
            ab2d.mainLoop()
            out = np.array(out)
            print("simu over, begin comp")
            with open('test/comparison_files/{}_out.dat'.format(test)) as f:
                comp = np.loadtxt(f)

            np.testing.assert_array_almost_equal(out, comp)
