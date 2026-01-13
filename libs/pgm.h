#ifndef PGM_H_
#define PGM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ppm.h"

typedef struct {
    int width, height, maxval;
    unsigned char *image;
} PGM;

PGM* read_pgm(const char* filename);

PPM* pgm_to_ppm(PGM* pgm_image);

PGM* pgm_make(int width, int height);

void pgm_free(PGM *pgm);

unsigned char pgm_gaussKernel(PGM *pgm, int x, int y, int MASK_SIZE, float *mask);

void pgm_gaussFilter(PGM *pgm, PGM *pgm_filtered, int MASK_SIZE, float SIGMA);

PGM* pgm_copy(PGM* pgm);

#endif
