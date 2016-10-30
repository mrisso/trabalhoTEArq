#define MALLOC_ERROR									1

#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMultKernel(float *dC, float *dA, float *dB, int width);
void popularMatriz(float *matriz, int tam, float valor);
float *matrixMultDevice(int width, int tileW);
int compareNum(float n1, float n2, float precisao);
void compareRes(float *mA, float *mB, int size);
