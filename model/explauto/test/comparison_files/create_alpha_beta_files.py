# The same file is generated if it is the same prefix
import hashlib
import subprocess
from io import BytesIO
import time

import numpy as np

def main(prefix):
    print ("Start gen")
    true_start = time.time()
    start = time.time()
    prefix_seed = int(hashlib.md5(prefix.encode("utf-8")).hexdigest(), 16) % 4294967295
    np.random.seed(prefix_seed)
    size = 70000
    params = np.random.normal(0, 1, (size, 2))
    params = params.cumsum(axis=0)
    params[:, 0] = np.abs(params[:, 0])
    np.savetxt("{}_ab.dat".format(prefix), params)
    print("end gen :", time.time() - start)

    print("begin synth")
    start = time.time()
    input_bytes = BytesIO()
    input_bytes.write(bytes(str(len(params)) + "\n", 'utf-8'))
    np.savetxt(input_bytes, params)
    print("true synth beg")
    synthstart = time.time()
    with open("{}_out.dat".format(prefix), "w") as outfile:
        subprocess.run(["../../../csynthesizer/alphabeta2dat"],
                        input=input_bytes.getvalue(),
                        stdout=outfile)
    input_bytes.close()
    print("end synth:", time.time() - start, "(synth time:", time.time() - synthstart, ")")

    print("start wav")
    start = time.time()
    subprocess.call(["../../../csynthesizer/dat2wav", "{}_out.dat".format(prefix),
                     "{}_wav.wav".format(prefix), "44100"])
    print("end wav:", time.time()-start)

if __name__ == "__main__":
    import sys

    main(sys.argv[1])
