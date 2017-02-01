import numpy as np
import subprocess

def extract_gte(wavfilename):
    out = subprocess.check_output(['../gteextractor/gteextractor.sh', wavfilename])
    res = np.fromstring(out, sep='\n', dtype=int)
    return res
