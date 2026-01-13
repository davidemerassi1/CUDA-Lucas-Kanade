#include <stdio.h>
#include <stdlib.h>
#include "pgm.h"
#include "ppm.h"

PGM* read_pgm(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("Errore apertura file"); return NULL; }

    //L'intestazione di un PGM è una riga <pgm_type, width, height, maxColor (in genere 255)>
    char format[3];
    int maxval;
    int width, height;
    fscanf(f, "%2s\n%d %d\n%d\n", format, &width, &height, &maxval);

    PGM *pgm = (PGM*)malloc(sizeof(PGM));
    pgm->width = width;
    pgm->height = height;
    pgm->maxval = maxval;
    pgm->image = (color*)malloc(width * height * sizeof(color));
    fread(pgm->image, 1, width * height, f);
    fclose(f);
    return pgm;
}

PPM* pgm_to_ppm(PGM* pgm_image) {
    pel bg = {0,0,0};
    PPM* ppm_image = ppm_make(pgm_image->width, pgm_image->height, bg);
    for(int y=0;y<pgm_image->height;y++){
        for(int x=0;x<pgm_image->width;x++){
            unsigned char g = pgm_image->image[y * pgm_image->width + x];
            pel c = {g, g, g};       // R=G=B = intensità del PGM
            ppm_set(ppm_image, x, y, c);
        }
    }
    return ppm_image;
}

PGM* pgm_make(int width, int height) {
    PGM* pgm = (PGM*)malloc(sizeof(PGM));
    if (!pgm) {
        perror("malloc PGM failed");
        return NULL;
    }

    pgm->width  = width;
    pgm->height = height;
    pgm->maxval = 255; // tipico per PGM
    pgm->image  = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!pgm->image) {
        perror("malloc PGM image failed");
        free(pgm);
        return NULL;
    }

    for (int i = 0; i < width * height; i++)
        pgm->image[i] = 0;

    return pgm;
}

unsigned char pgm_gaussKernel(PGM *pgm, int x, int y, int MASK_SIZE, float *mask) {
    float sum = 0;
    int RADIUS = MASK_SIZE / 2;
    for (int r = 0; r < MASK_SIZE; r++) {
        for (int c = 0; c < MASK_SIZE; c++) {
            int row = y + r - RADIUS;
            int col = x + c - RADIUS;
            if (row >= 0 && row < pgm->height && col >= 0 && col < pgm->width) {
                float m = mask[r * MASK_SIZE + c];
                unsigned char p = pgm->image[row * pgm->width + col];
                sum += p * m;
            }
        }
    }
    // clamp
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    return (unsigned char)sum;
}

void pgm_gaussFilter(PGM *pgm, PGM *pgm_filtered, int MASK_SIZE, float SIGMA) {
    float *mask = gaussMask(MASK_SIZE, SIGMA);
    for (int y = 0; y < pgm->height; y++) {
        for (int x = 0; x < pgm->width; x++) {
            pgm_filtered->image[y * pgm->width + x] = pgm_gaussKernel(pgm, x, y, MASK_SIZE, mask);
        }
    }
    free(mask);
}

void pgm_free(PGM *pgm) {
    if (pgm == NULL) return;
    if (pgm->image != NULL) {
        free(pgm->image);
        pgm->image = NULL;
    }
    free(pgm);
}

PGM* pgm_copy(PGM* pgm) {
    PGM *pgm1 = (PGM *)malloc(sizeof(PGM));
    pgm1->image = (unsigned char *)malloc(pgm->width * pgm->height);
    pgm1->width = pgm->width;
    pgm1->height = pgm->height;
    pgm1->maxval = pgm->maxval;
    memcpy(pgm1->image, pgm->image, pgm->width * pgm->height);
    return pgm1;
}
