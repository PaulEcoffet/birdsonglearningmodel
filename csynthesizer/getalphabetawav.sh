#! /bin/bash
first_dir="$(pwd)"
dir=$(dirname "$(realpath "$0")")


fname=$(basename $1 | cut -d. -f1)
cp "$1" "$dir/gabwtmp.wav"

cd "$(dirname "$(realpath "$0")")";

echo "converting to ascii wav"

./wta gabwtmp.wav > gabwtmp.dat

echo "getting fundamental frequencies"
nbinfo=$(cat gabwtmp.dat | wc -l)
./computeFF $nbinfo gabwtmp.dat > gabwtmp_FF.dat
./smoothFF gabwtmp_FF.dat > gabwtmp_smooth.dat

echo "synthesize"
./synthesize gabwtmp_smooth.dat gabwtmp_out.dat env_gabwtmp.dat

echo "gather a and b"
python - << EOF
import numpy as np
alpha = np.loadtxt('env_gabwtmp.dat')[:, 2]
beta = np.loadtxt('gabwtmp_out.dat')[:, 1]
np.savetxt('gabwtmp_ab.dat', np.stack((alpha[:beta.shape[0]], beta), axis=-1))

EOF
echo $(cat gabwtmp_ab.dat | wc -l) | cat - gabwtmp_ab.dat > gabwtmp_ab_num.dat

./dat2wav song_gabwtmp_out.dat gabwtmp_out.wav 44100

mv gabwtmp_ab.dat "$first_dir/${fname}_ab.dat"
mv song_gabwtmp_out.dat "$first_dir/${fname}_out.dat"
mv gabwtmp_ab_num.dat "$first_dir/${fname}_out_num.dat"
mv gabwtmp_out.wav "$first_dir/${fname}_out.wav"

rm gabwtmp*
