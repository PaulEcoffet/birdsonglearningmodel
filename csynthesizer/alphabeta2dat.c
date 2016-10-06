/*
  Written by Paul Ecoffet, using the code from the DYNAMICAL SYSTEMS LAB
  Physics Department, Universidad de Buenos Aires
 */

#include <stdlib.h>
#include <stdio.h>

int countlines(char *filename) {
  char ch;
  int nb_lines = 0;
  FILE *fp = fopen(filename, "r");
  while ((ch = fgetc(fp)) != EOF) {
    if (ch == '\n') {
      nb_lines++;
    }
  }
  fclose(fp);
  return nb_lines;
}

int main(int argc, char *argv[]) {
  int size = 0;
  if (argc < 3) {
    fprintf(stderr, "Please provide the salida file name and the envelope file name.\n");
    return 1;
  }
  size = countlines(argv[1]);
  finch(size, argv[2], argv[1]);

  return 0;
}
