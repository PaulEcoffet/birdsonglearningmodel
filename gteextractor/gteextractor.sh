#! /bin/bash
first_dir="$(pwd)"
dir=$(dirname "$(realpath "$0")")

fname=$(basename $1 | cut -d. -f1)
cp "$1" "$dir/workon.wav"

cd "$(dirname "$(realpath "$0")")";

./wta workon.wav > workon.dat


./gtes_example workon.dat > /dev/null 2>&1


cat gtes1.workon.dat gtes2.workon.dat gtes3.workon.dat gtes4.workon.dat | sort -n | uniq | sed /^0/d



rm workon.wav
rm workon.dat
rm gtes1.workon.dat gtes2.workon.dat gtes3.workon.dat gtes4.workon.dat
