#define DEBUG 					0
#define ERRO_N_ARGS				1
#define OK						0

#include "seqMatrix.h"

int main (int argc, char *argv[])
{
	if(argc!=3){
		printf("Modo de uso: %s <tamanho-da-matriz> <numero-de-repeticoes>\n",argv[0]);
		return ERRO_N_ARGS;
	}

	int i;
	int tam = atoi(argv[1]);
	int rep = atoi(argv[2]);

	double * matA = (double*)malloc(tam*tam*sizeof(double));
	double * matB = (double*)malloc(tam*tam*sizeof(double));
	double * matriz;
	
	for(i=0;i<tam*tam;i++)
	{
		matA[i] = 1.0f;//((double)i+5)/7;
	}

	for(i=0;i<tam*tam;i++)
	{
		matB[i] = 0.01f;//((double)i+4)/3;
	}
	
	matriz = multiplicaMatriz(matA, matB, tam, rep);
	if(DEBUG)
		for(i=0;i<tam*tam;i++){
			if((i%tam)==0 && i>0)
				printf("\n\t%.2f",matriz[i]);
			else
				printf("\t%.2f",matriz[i]);
		}
	free(matA);
	free(matB);
	free(matriz);
	return OK;
}
