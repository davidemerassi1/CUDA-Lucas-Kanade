# CUDA-Lucas-Kanade
Compilare ed eseguire con 
```
!nvcc -arch=sm_75  src/progetto/lucaskanade.cu -o lucaskanade -I libs libs/ppm.cpp libs/pgm.cpp
!./lucaskanade
```