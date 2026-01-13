
#include <stdio.h>
#include <stdlib.h>
#include "libs/ppm.h"
#include "libs/pgm.h"

#define max_features 1000

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

    // Calcoliamo i pesi (quanto siamo lontani dai bordi del pixel)
    float dx = x - (float)x1;
    float dy = y - (float)y1;

    // Recuperiamo i valori dei 4 pixel vicini
    // Nota: bisognerebbe controllare che x2 e y2 non escano dai bordi dell'immagine
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
    int window_size = 11; // Una finestra 11x11 è ottima per gestire pattern puntinati
    int half = window_size / 2;
    int max_iterations = 10;

    for (int f = 0; f < num_features; f++) {
        if (!features[f].active) continue;

        float u = 0, v = 0; // Spostamento iniziale (ipotesi di zero movimento)
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

                    // Coordinate nel secondo frame (traslate dallo shift u,v)
                    float nx = px + u;
                    float ny = py + v;

                    // Controllo bordi per evitare crash con l'interpolazione
                    if (nx < 1 || nx >= img2->width - 1 || ny < 1 || ny >= img2->height - 1) continue;

                    // 1. Calcolo gradienti spaziali (su frame 1)
                    float Ix = (get_pixel_bilinear(img1, px + 1, py) - get_pixel_bilinear(img1, px - 1, py)) * 0.5f;
                    float Iy = (get_pixel_bilinear(img1, px, py + 1) - get_pixel_bilinear(img1, px, py - 1)) * 0.5f;

                    // 2. Calcolo differenza temporale (I2_nuovo - I1_vecchio)
                    float It = get_pixel_bilinear(img2, nx, ny) - get_pixel_bilinear(img1, px, py);
                    sumIx2 += Ix * Ix;
                    sumIy2 += Iy * Iy;
                    sumIxIy += Ix * Iy;
                    sumIxIt += Ix * It;
                    sumIyIt += Iy * It;
                }
            }

            // 3. Risoluzione del sistema lineare 2x2 (Regola di Cramer)
            float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;

            if (fabs(det) < 0.001f) { 
                features[f].active = 0; // Punto perso: zona troppo piatta o ambigua
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
    scale_features(h_features, num_features, 2);
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
      //Nella versione finale bisogna usare blur su gpu o piramidi di gauss
      //oppure fare blur solo intorno a ciascuna feature
      PGM* frame2_downsampled = pgm_make(frame2->width/2, frame2->height/2);
      downsample_half(frame2, frame2_downsampled);
      PGM* frame2_blur = pgm_make(frame2_downsampled->width, frame2_downsampled->height);
      pgm_gaussFilter(frame2_downsampled, frame2_blur, KERNEL_SIZE, SIGMA);
      //PGM* frame2_blur = pgm_copy(frame2);
      scale_features(h_features, num_features, 0.5);
      track_features(frame1_blur, frame2_blur, h_features, num_features);
      scale_features(h_features, num_features, 2);
      PPM* out = pgm_to_ppm(frame2);
      draw_features(out, h_features, num_features);
      ppm_write(out, make_output_filename(t));

      pgm_free(frame1);
      pgm_free(frame1_downsampled);
      pgm_free(frame1_blur);
      frame1 = frame2;
      frame1_downsampled = frame2_downsampled;
      frame1_blur = frame2_blur;
      ppm_free(out);
      t++;
    }

    return 0;
}
