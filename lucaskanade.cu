
#include <stdio.h>
#include <stdlib.h>
#include "libs/ppm.h"
#include "libs/pgm.h"

#define max_features 1000

typedef struct {
    float x;
    float y;
    int active;
} Feature;

typedef struct {
    float val;      // Lo score di Shi-Tomasi
    Feature feat;
} ScoredFeature;

void downsample_half(PGM *src, PGM *dst) {
    for (int y = 0; y < dst->height; y++) {
        for (int x = 0; x < dst->width; x++) {
            // Media 2x2 per evitare aliasing (rumore)
            int sum = src->image[(2*y) * src->width + (2*x)] +
                      src->image[(2*y) * src->width + (2*x+1)] +
                      src->image[(2*y+1) * src->width + (2*x)] +
                      src->image[(2*y+1) * src->width + (2*x+1)];
            dst->image[y * dst->width + x] = (unsigned char)(sum / 4);
        }
    }
}

void redistribute_features(Feature* features, int numFeatures) {
    for (int i = 0; i < numFeatures; i++) {
      features[i].x *= 2;
      features[i].y *= 2;
    }
}

bool is_too_close(int x1, int x2, int y1, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    return dx*dx + dy*dy < 100;
}
        
int compare_scored_features(const void *a, const void *b) {
    ScoredFeature *f1 = (ScoredFeature *)a;
    ScoredFeature *f2 = (ScoredFeature *)b;
    if (f1->val < f2->val) return 1;
    if (f1->val > f2->val) return -1;
    return 0;
}

float compute_shi_tomasi_score(PGM *pgm, int x, int y, int window_size) {
    float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0;
    int half = window_size / 2;
    int w = pgm->width;
    unsigned char *img = pgm->image;

    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            int px = x + j;
            int py = y + i;

            //gradiente x
            float Ix =
                -1 * img[(py-1)*w + (px-1)] + 1 * img[(py-1)*w + (px+1)] +
                -2 * img[(py)*w   + (px-1)] + 2 * img[(py)*w   + (px+1)] +
                -1 * img[(py+1)*w + (px-1)] + 1 * img[(py+1)*w + (px+1)];

            //gradiente y
            float Iy =
                -1 * img[(py-1)*w + (px-1)] - 2 * img[(py-1)*w + px] - 1 * img[(py-1)*w + (px+1)] +
                 1 * img[(py+1)*w + (px-1)] + 2 * img[(py+1)*w + px] + 1 * img[(py+1)*w + (px+1)];

            sumIx2 += Ix * Ix;
            sumIy2 += Iy * Iy;
            sumIxIy += Ix * Iy;
        }
    }

    float trace = sumIx2 + sumIy2;
    float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
    float term = sqrtf((trace * trace / 4.0f) - det);
    float lambda2 = (trace / 2.0f) - term;

    return lambda2;
}

int find_good_features(PGM* pgm, Feature* features) {
    ScoredFeature *scored_features = (ScoredFeature*) malloc(pgm->height * pgm->width * sizeof(ScoredFeature));
    int count=0;

    for (int y = 5; y < pgm->height - 5; y++) {
        for (int x = 5; x < pgm->width - 5; x++) {
            float s = compute_shi_tomasi_score(pgm, x, y, 3);
            if (s > 40000) { // Soglia minima di qualità
                scored_features[count].val = s;
                scored_features[count].feat.x = (float)x;
                scored_features[count].feat.y = (float)y;
                scored_features[count].feat.active = 1;
                count++;
            }
        }
    }

    qsort(scored_features, count, sizeof(ScoredFeature), compare_scored_features);

    int selected_feats = 0;
    for (int i=0; i<count && selected_feats<max_features; i++) {
        bool too_close = false;
        for (int z = 0; z < selected_feats && !too_close; z++) {
            too_close = is_too_close(scored_features[i].feat.x, features[z].x, scored_features[i].feat.y, features[z].y);
        }
        if (!too_close) {
            features[selected_feats] = scored_features[i].feat;
            selected_feats++;
        }
    }

    return selected_feats;
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
        if (features[i].active)
          draw_feature(image, (int)features[i].x, (int)features[i].y, 5, (pel){255,0,0});
    }
}

char* make_input_filename(int frame_index) {
    char *result = (char*)malloc(64);
    sprintf(result, "frames/frame_%04d.pgm", frame_index);
    return result;
}

char* make_output_filename(int frame_index) {
    char *result = (char*)malloc(64);
    sprintf(result, "result/frame_%04d.ppm", frame_index);
    return result;
}

int main(void) {
    PGM* frame1 = read_pgm(make_input_filename(1));
    PGM* frame1_downsampled = pgm_make(frame1->width/2, frame1->height/2);
    downsample_half(frame1, frame1_downsampled);
    PGM* frame1_blur = pgm_make(frame1_downsampled->width, frame1_downsampled->height);
    int KERNEL_SIZE = 5;   // dimensione finestra
    float SIGMA = 1.2;    // deviazione standard
    pgm_gaussFilter(frame1_downsampled, frame1_blur, KERNEL_SIZE, SIGMA);
    Feature* h_features = (Feature*)malloc(max_features*sizeof(Feature));
    int num_features = find_good_features(frame1_blur, h_features);
    redistribute_features(h_features, num_features);
    PPM* converted = pgm_to_ppm(frame1);
    draw_features(converted, h_features, num_features);
    ppm_write(converted, make_output_filename(1));

    return 0;
}
