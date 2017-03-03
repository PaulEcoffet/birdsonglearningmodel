#!/bin/sh
# Simply build alphabeta2dat, which takes a stream of alphabeta prefixed with
# the size of the stream (number of lines) and output the signal
# (which should be normalized)

gcc alphabeta2dat.c rk4.c finch_void_stdin.c -lm -o alphabeta2dat
