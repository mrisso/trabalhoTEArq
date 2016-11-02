#ifndef _MATRIX_MULT_
#define _MATRIX_MULT_

#define MALLOC_ERROR									1
#define DEBUG											0

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern "C"{
#include "../../../sequential/double/seqMatrix.h"
}

__global__ void matrixMultKernel(double *dC, double *dA, double *dB, long width);
void popularMatriz(double *matriz, int tam, double valor);
double *matrixMultDevice(int width, int rep);
int compareNum(double n1, double n2, double precisao);
int compareRes(double *mA, double *mB, int size);

#endif
