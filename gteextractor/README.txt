%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Automatic GTE extraction %
%      September 2015      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

README and instructions.

This program uses the algorithm described in ‘Automatic reconstruction of physiological gestures used in a model of birdsong production’ by Santiago Boari, Yonatan Sanz Perl, Ana Amador, Daniel Margoliash and Gabriel B. Mindlin. 

The following instructions describe the compiling, usage and outputs of the program.

Files required for analysis: 
- A song recording, sampled at 40 kHz (i.e.: exampleGTE.wav)

1. Convert from WAV file to .dat ASCII
Note that you can convert to .dat on your own, but here we provide a code that does that. 
- Compile ‘wta.c’ (wav-to-ascii) in terminal, by typing:
cc wta.c -lm -o wta
- Run it by giving the input file and the output file name:
./wta exampleGTE.wav > exampleGTE.dat
Please note that you can convert from .WAV to ASCII using your own software. If you do so, please name your dat file ‘exampleGTE.dat’. You can adjust this to any other name by editing the name of the input files in ’gtes_example.c’

2. Compile the automatic GTE detection program, by typing:
gcc gtes_example.c -lm -o gtes_example

3. Run it, simply by typing: ./gtes_example
The program will automatically detect the required files, provided you followed previous steps.
This program creates separate files for each GTE type, computed from the sound envelope:
- gtes1.X.dat : syllable beginnings/ends
- gtes2.X.dat : absolute maximum of each syllable
- gtes3.X.dat : intrasyllabic significant minima (see paper for details)
- gtes4.X.dat : last maximum of each syllable

where X is the name of your original file (i.e., exampleGTE in this case).
The output in these files is the timestamp of each GTE, in number of samples.
To get GTE times in miliseconds, divide by 40 (sampling freq.: 40 kHz);


