# The same file is generated if it is the same prefix
import hashlib
import subprocess

import numpy as np

def main(prefix):
    prefix_seed = int(hashlib.md5(prefix.encode("utf-8")).hexdigest(), 16) % 4294967295
    np.random.seed(prefix_seed)
    size = np.random.randint(45000, 70000)
    params = np.zeros((size, 2))
    params[0] = np.random.normal(0, 1, 2)
    for i in range(1, size):
        params[i] = params[i-1] + np.random.normal(0, 0.5, 2)
    params[:, 0] = np.abs(params[:, 0])

    with open("{}_alpha.dat".format(prefix), "w") as alphaf:
        with open("{}_beta.dat".format(prefix), "w") as betaf:
            for i in range(size):
                alphaf.write("0 0 {}\n".format(params[i, 0])) # padding to respect the Buenos Aires format
                betaf.write("0 {}\n".format(params[i, 1]))

    with open("{}_out.dat".format(prefix), "w") as out:
        subprocess.call(["../../csynthesizer/alphabeta2dat",
                         "{}_alpha.dat".format(prefix),
                         "{}_beta.dat".format(prefix)],
                        stdout=out)

    subprocess.call(["../../csynthesizer/dat2wav", "{}_out.dat".format(prefix),
                     "{}_wav.wav".format(prefix), "44100"])

if __name__ == "__main__":
    import sys

    main(sys.argv[1])
