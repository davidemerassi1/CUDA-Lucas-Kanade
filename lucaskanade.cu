
#include <stdio.h>
#include <stdlib.h>
#include "../../GPUcomputing/utils/PPM/ppm.h"


unsigned char* read_pgm(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("Errore apertura file"); return NULL; }

    //L'intestazione di un PGM è una riga <pgm_type, width, height, maxColor (in genere 255)>
    char magic[3];
    int maxval;
    fscanf(f, "%2s\n%d %d\n%d\n", magic, width, height, &maxval);

    unsigned char* img = (unsigned char*)malloc((*width) * (*height));
    fread(img, 1, (*width) * (*height), f);
    fclose(f);
    return img;
}

PPM* pgm_to_ppm(const unsigned char* pgm_image, int width, int height) {
    pel bg = {0,0,0};
    PPM* ppm_image = ppm_make(width, height, bg);
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            unsigned char g = pgm_image[y * width + x];
            pel c = {g, g, g};       // R=G=B = intensità del PGM
            ppm_set(ppm_image, x, y, c);
        }
    }
    return ppm_image;
}

int main(void) {
    int width, height;
    unsigned char* img = read_pgm("frames/frame_0001.pgm", &width, &height);
    PPM* converted = pgm_to_ppm(img, width, height);
    ppm_write(converted, "frame0001.ppm");
        
    return 0;
}
