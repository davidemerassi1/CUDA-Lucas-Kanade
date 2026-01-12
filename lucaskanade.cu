
#include <stdio.h>
#include <stdlib.h>
#include "../../GPUcomputing/utils/PPM/ppm.h"

#define max_features 1000

typedef struct {
    int width, height, maxval;
    unsigned char *image;
} PGM;

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

    // opzionale: inizializza a zero
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

typedef struct {
    int x;
    int y;
    bool active;
} Feature;

//sarà da rendere gpu
int detect_features(PGM* pgm, Feature* features) {
    int count = 0;
    int THRESHOLD = 35;
    for(int y=10; y<pgm->height-10 && count<max_features; y+=4){
        for(int x=10; x<pgm->width-10 && count<max_features; x+=4){
            //calcolo del gradiente
            int gx = abs(pgm->image[y*pgm->width + (x+1)] - pgm->image[y*pgm->width + (x-1)]);
            int gy = abs(pgm->image[(y+1)*pgm->width + x] - pgm->image[(y-1)*pgm->width + x]);

            if (gx > THRESHOLD && gy > THRESHOLD) {
                features[count].x = x;
                features[count].y = y;
                features[count].active = 1;
                count++;
            }
        }
    }
    return count;
}

void draw_feature(PPM* image, int cx, int cy, int radius, pel col) {
    for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {

            if (dx*dx + dy*dy <= radius*radius) {
                int x = cx + dx;
                int y = cy + dy;

                if (x >= 0 && x < image->width &&
                    y >= 0 && y < image->height) {
                    ppm_set(image, x, y, col);
                }
            }
        }
    }
}

void draw_features(PPM* image, Feature* features, int num_features) {
    for(int i=0; i<num_features; i++){
        draw_feature(image, features[i].x, features[i].y, 5, (pel){255,0,0});
    }
}

int main(void) {
    PGM* frame1 = read_pgm("frames/frame_0001.pgm");
    PGM* frame1_blur = pgm_make(frame1->width, frame1->height);
    int KERNEL_SIZE = 10;   // dimensione finestra
    float SIGMA = 1.2;    // deviazione standard
    pgm_gaussFilter(frame1, frame1_blur, KERNEL_SIZE, SIGMA);
    Feature* h_features = (Feature*)malloc(max_features*sizeof(Feature));
    int num_features = detect_features(frame1_blur, h_features);
    PPM* converted = pgm_to_ppm(frame1);
    draw_features(converted, h_features, num_features);
    ppm_write(converted, "frame0001.ppm");

    return 0;
}
