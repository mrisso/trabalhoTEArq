#ifndef _MATRIX_MULT_
#define _MATRIX_MULT_

#define MALLOC_ERROR									1
#define DEBUG											0

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern "C"{
#include "../../../sequential/float/seqMatrix.h"
}

__global__ void matrixMultKernel(float *dC, float *dA, float *dB, long width);
void popularMatriz(float *matriz, int tam, float valor);
float *matrixMultDevice(int width, int rep);
int compareNum(float n1, float n2, float precisao);
int compareRes(float *mA, float *mB, int size);

#endif
