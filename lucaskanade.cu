
#include <stdio.h>
#include <stdlib.h>
#include "libs/ppm.h"
#include "libs/pgm.h"

#define MAX_FEATURES 1000
#define PYRAMID_LEVELS 3
#define FIRST_ANALYSIS_LEVEL 2

typedef struct {
    float x;
    float y;
    int active;
    pel feat_color;
} Feature;

typedef struct {
    float val;      // Lo score di Shi-Tomasi
    Feature feat;
} ScoredFeature;

pel colors[10] = {
        {255, 0, 0},   // Rosso
        {0, 255, 0},   // Verde
        {0, 0, 255},   // Blu
        {255, 255, 0}, // Giallo
        {255, 255, 255}, // Bianco
        {0, 0, 0},     // Nero
        {128, 128, 128}, // Grigio
        {255, 165, 0}, // Arancione
        {75, 0, 130},  // Indaco
        {238, 130, 238} // Violetto
    };

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

void scale_features(Feature* features, int numFeatures, float multiplier) {
    for (int i = 0; i < numFeatures; i++) {
      features[i].x *= multiplier;
      features[i].y *= multiplier;
    }
}

bool is_too_close(int x1, int x2, int y1, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    return dx*dx + dy*dy < (200 / pow(2, FIRST_ANALYSIS_LEVEL));
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

int find_good_features(PGM* pgm, Feature* features, int max_features) {
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
                scored_features[count].feat.feat_color = colors[count%10];
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
          draw_feature(image, (int)features[i].x, (int)features[i].y, 5, features[i].feat_color);
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

float get_pixel_bilinear(PGM *img, float x, float y) {
    int x1 = (int)x;
    int y1 = (int)y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float dx = x - (float)x1;
    float dy = y - (float)y1;

    unsigned char p11 = img->image[y1 * img->width + x1];
    unsigned char p21 = img->image[y1 * img->width + x2];
    unsigned char p12 = img->image[y2 * img->width + x1];
    unsigned char p22 = img->image[y2 * img->width + x2];

    // Formula dell'interpolazione bilineare
    float val = (1.0f - dx) * (1.0f - dy) * p11 +
                dx * (1.0f - dy) * p21 +
                (1.0f - dx) * dy * p12 +
                dx * dy * p22;

    return val;
}

void track_features(PGM *img1, PGM *img2, Feature *features, int num_features) {
    int window_size = 11;
    int half = window_size / 2;
    int max_iterations = 10;

    for (int f = 0; f < num_features; f++) {
        if (!features[f].active) continue;

        float u = 0, v = 0;
        float curr_x = features[f].x;
        float curr_y = features[f].y;

        // Loop di raffinamento Newton-Raphson
        for (int iter = 0; iter < max_iterations; iter++) {
            float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0;
            float sumIxIt = 0, sumIyIt = 0;

            for (int i = -half; i <= half; i++) {
                for (int j = -half; j <= half; j++) {
                    float px = curr_x + j;
                    float py = curr_y + i;

                    float nx = px + u;
                    float ny = py + v;

                    if (nx < 1 || nx >= img2->width - 1 || ny < 1 || ny >= img2->height - 1) continue;

                    float Ix = (get_pixel_bilinear(img1, px + 1, py) - get_pixel_bilinear(img1, px - 1, py)) * 0.5f;
                    float Iy = (get_pixel_bilinear(img1, px, py + 1) - get_pixel_bilinear(img1, px, py - 1)) * 0.5f;

                    float It = get_pixel_bilinear(img2, nx, ny) - get_pixel_bilinear(img1, px, py);
                    sumIx2 += Ix * Ix;
                    sumIy2 += Iy * Iy;
                    sumIxIy += Ix * Iy;
                    sumIxIt += Ix * It;
                    sumIyIt += Iy * It;
                }
            }

            // Regola di Cramer
            float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;

            if (fabs(det) < 0.001f) {
                features[f].active = 0;
                break;
            }

            float du = (sumIxIy * sumIyIt - sumIy2 * sumIxIt) / det;
            float dv = (sumIxIy * sumIxIt - sumIx2 * sumIyIt) / det;

            u += du;
            v += dv;

            // Se lo spostamento è quasi nullo, abbiamo convergenza
            if (du * du + dv * dv < 0.005f) break;
        }

        // Aggiorniamo la posizione finale della feature
        features[f].x += u;
        features[f].y += v;

        // Controllo finale: se la feature è uscita dall'immagine, disattivala
        if (features[f].x < 0 || features[f].x >= img2->width ||
            features[f].y < 0 || features[f].y >= img2->height) {
            features[f].active = 0;
        }
    }
}

void build_pyramid(PGM* frame, PGM** pyramid, int levels) {
    // Il livello 0 è il frame originale
    pyramid[0] = pgm_copy(frame);
    for (int l = 1; l < levels; l++) {
        //ogni livello successivo è una versione grande la metà e con un blur (eseguito prima di scalare)
        PGM* blurred_prev = pgm_make(pyramid[l-1]->width, pyramid[l-1]->height);
        pgm_gaussFilter(pyramid[l-1], blurred_prev, 5, 1.0);
        pyramid[l] = pgm_make(pyramid[l-1]->width / 2, pyramid[l-1]->height / 2);
        downsample_half(blurred_prev, pyramid[l]);
        pgm_free(blurred_prev);
    }
}

void track_features_pyramidal(PGM** pyr1, PGM** pyr2, Feature* feats, int n) {
    //scale = 1 / (2^(l-1))
    float initial_scale = 1.0f / powf(2.0f, PYRAMID_LEVELS - 1);
    scale_features(feats, n, initial_scale);
    for (int l = PYRAMID_LEVELS - 1; l >= 0; l--) {
        track_features(pyr1[l], pyr2[l], feats, n);
        if (l > 0) {
            scale_features(feats, n, 2.0f);
        }
    }
}

int main(void) {
    PGM* frame1 = read_pgm(make_input_filename(1));
    PGM* frame1_pyramid[PYRAMID_LEVELS]; //il primo livello equivarrà a frame1
    build_pyramid(frame1, frame1_pyramid, PYRAMID_LEVELS);
    Feature* h_features = (Feature*)malloc(MAX_FEATURES*sizeof(Feature));
    //Vengono cercate le feature sul primo frame. Conviene usare la versione già filtrata 
    int num_features = find_good_features(frame1_pyramid[FIRST_ANALYSIS_LEVEL], h_features, MAX_FEATURES);
    scale_features(h_features, num_features, pow(2, FIRST_ANALYSIS_LEVEL));
    PPM* converted = pgm_to_ppm(frame1);
    draw_features(converted, h_features, num_features);
    ppm_write(converted, make_output_filename(1));
    ppm_free(converted);
    PGM* frame2;
    int t = 2;
    while (true) {
      frame2 = read_pgm(make_input_filename(t));
      if (frame2 == NULL)
          break;
      PGM* frame2_pyramid[PYRAMID_LEVELS];
      //Generazione della piramide di Gauss per i frame successivi (blur + scalatura)
      build_pyramid(frame2, frame2_pyramid, PYRAMID_LEVELS);
      //PGM* frame2_pyramid = pgm_copy(frame2);
      //tracciamento con Lucas-Kanade
      track_features_pyramidal(frame1_pyramid, frame2_pyramid, h_features, num_features);
      PPM* out = pgm_to_ppm(frame2);
      draw_features(out, h_features, num_features);
      ppm_write(out, make_output_filename(t));
      ppm_free(out);
      for(int l=0; l < PYRAMID_LEVELS; l++) {
        pgm_free(frame1_pyramid[l]);
        frame1_pyramid[l] = frame2_pyramid[l];
      }
      frame1 = frame2;
      t++;
    }
    return 0;
}
