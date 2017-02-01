#! /bin/bash
first_dir="$(pwd)"
dir=$(dirname "$(realpath "$0")")

fname=$(basename $1 | cut -d. -f1)
cp "$1" "$dir/workon.wav"

cd "$(dirname "$(realpath "$0")")";

echo "converting to ascii wav"

./wta workon.wav > workon.dat

echo "extracting GTE"

./gtes_example workon.dat

echo "concatening files"

cat gtes1.workon.dat gtes2.workon.dat gtes3.workon.dat gtes4.workon.dat | sort -n | sed /^0/d > gte_workon.dat

echo "clean everything up"

mv gte_workon.dat "$first_dir/${fname}_gte.dat"

rm workon.wav
rm workon.dat
rm gtes1.workon.dat gtes2.workon.dat gtes3.workon.dat gtes4.workon.dat

echo "done"
