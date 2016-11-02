#define DEBUG 					0
#define ERRO_N_ARGS				1
#define OK						0

#include "ompMatrix.h"
#include <time.h>

void freeMatrix(float **m, int width){
	for(int i=0;i<width;i++)
		free(m[i]);
	free(m);
}

int main (int argc, char *argv[])
{
	if(argc!=3){
		printf("Modo de uso: %s <tamanho-da-matriz> <numero-de-repeticoes>\n",argv[0]);
		return ERRO_N_ARGS;
	}

	int i,j;
	int tam = atoi(argv[1]);
	int rep = atoi(argv[2]);

	float **matA = (float **)malloc(tam * sizeof(float*));
	for(i = 0; i < tam; i++) matA[i] = (float *)malloc(tam * sizeof(float));
	float **matB = (float **)malloc(tam * sizeof(float*));
	for(i = 0; i < tam; i++) matB[i] = (float *)malloc(tam * sizeof(float));
	float **matriz = (float **)malloc(tam * sizeof(float*));
	for(i = 0; i < tam; i++) matriz[i] = (float *)malloc(tam * sizeof(float));

	for(i=0;i<tam;i++)
	{
		for(j=0;j<tam;j++)
			matA[i][j] = 1.0f;//((float)i+5)/7;
	}

	for(i=0;i<tam;i++)
	{
		for(j=0;j<tam;j++)
			matB[i][j] = 0.01f;//((float)i+5)/7;
	}

	time_t t;
	t = clock();
	
#pragma omp parallel for
	for(i=0;i<rep;i++)
		ompMultMatrix(matA,matB,matriz,tam);

	t = clock() - t;
	double tempoExe = ((double)t)/CLOCKS_PER_SEC;
	printf("%lf\n\n",tempoExe);

	freeMatrix(matA,tam);
	freeMatrix(matB,tam);
	freeMatrix(matriz,tam);
	return OK;
}
