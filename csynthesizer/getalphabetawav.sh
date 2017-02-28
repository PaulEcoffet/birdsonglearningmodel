#! /bin/bash
first_dir="$(pwd)"
dir=$(dirname "$(realpath "$0")")


fname=$(basename $1 | cut -d. -f1)
cp "$1" "$dir/gabwtmp.wav"

cd "$(dirname "$(realpath "$0")")";

echo "converting to ascii wav"

./wta gabwtmp.wav > gabwtmp.dat

# Prepend with silence because otherwise the synth ignore the very first
# and very last sounds.
cat silence.dat gabwtmp.dat silence.dat > gabwtmp_prep.dat

refsonglength=$(cat gabwtmp.dat | wc -l)

echo "Length of the first signal $refsonglength"

echo "getting fundamental frequencies"
nbinfo=$(cat gabwtmp_prep.dat | wc -l)
./computeFF $nbinfo gabwtmp_prep.dat > gabwtmp_FF.dat
./smoothFF gabwtmp_FF.dat > gabwtmp_smooth.dat

echo "synthesize"
./synthesize gabwtmp_smooth.dat gabwtmp_out.dat env_gabwtmp_prep.dat

echo "gather a and b"
python - << EOF
import numpy as np
alpha = np.loadtxt('env_gabwtmp_prep.dat')[:, 2]
beta = np.loadtxt('gabwtmp_out.dat')[:, 1]
out = np.stack((alpha[:beta.shape[0]], beta), axis=-1)
np.savetxt('gabwtmp_ab.dat', out[1000:-1000])

EOF
echo $(cat gabwtmp_ab.dat | wc -l) | cat - gabwtmp_ab.dat > gabwtmp_ab_num.dat

# Now, let's remove the zeros we have put in front and at the end of the wave
# We take the data so it's the exact same length


tail -n +1001 song_gabwtmp_out.dat | head -n $refsonglength > song_gabwtmp_out_t.dat

echo "Length of out signal"
echo `cat song_gabwtmp_out_t.dat | wc -l`

echo "Converting to WAV file"

python - << EOF
from scipy.io import wavfile
import numpy as np

signal = np.loadtxt('song_gabwtmp_out_t.dat', dtype=float)
signal -= np.mean(signal)
signal = 2*(signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
wavfile.write('gabwtmp_out.wav', 44100, signal)

signal = np.loadtxt('gabwtmp.dat', dtype=float)
signal -= np.mean(signal)
signal = 2*(signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
wavfile.write('gabwtmp.wav', 44100, signal)
EOF

mv gabwtmp_ab.dat "$first_dir/${fname}_ab.dat"
mv song_gabwtmp_out_t.dat "$first_dir/${fname}_out.dat"
mv gabwtmp_ab_num.dat "$first_dir/${fname}_out_num.dat"
mv gabwtmp_out.wav "$first_dir/${fname}_out.wav"

rm *gabwtmp*
