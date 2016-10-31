#define DEBUG 					0
#define ERRO_N_ARGS				1
#define OK						0

#include <stdio.h>
#include <stdlib.h>

float * MultiplicaMatriz(float * matA, float * matB, int tam, int rep) //Função recebe 2 matrizes (matA e matB), tamanho de uma dimensão da matriz quadrada (tam) e quantas repetições da mesma multiplicação serão feitas (rep).//
{
	int i,j,k,lugar,r;
	/*
	 * i, j e k são contadores para fors do algoritmo de multiplicação
	 * lugar é o indice da matriz resultado (matriz)
	 * r é o contador do for de repetições
	 */
	float* matriz = (float*)malloc((tam*tam)*(sizeof(float)));
	float cValue = 0;
	
	/* matriz é a matriz resultado que queremos que seja a multiplicação de
	 * matA por matB nesta ordem.
	 */

	lugar = 0;
	
		for(r = 0; r<rep; r++)//Este for serve para repetir a multiplicação rep vezes afim de calcular o tempo para comparação com o CUDA
		for(i=0;i<tam;i++)
		{
			for(j=0;j<tam;j++){
				for(k=0;k<tam;k++)
				{
					cValue+=matA[i*tam+k] * matB[k*tam+j];
				}
				matriz[lugar]=cValue;
				lugar++;
				cValue = 0;
			}
		}
	return matriz;
}

int main (int argc, char *argv[])
{
	if(argc!=3){
		printf("Modo de uso: %s <tamanho-da-matriz> <numero-de-repeticoes>\n",argv[0]);
		return ERRO_N_ARGS;
	}

	int i;
	int tam = atoi(argv[1]);
	int rep = atoi(argv[2]);

	float * matA = (float*)malloc(tam*tam*sizeof(float));
	float * matB = (float*)malloc(tam*tam*sizeof(float));
	float * matriz;// = (float*)malloc(tam*tam*sizeof(float));
	
	for(i=0;i<tam*tam;i++)
	{
		matA[i] = 1.0f;//((float)i+5)/7;
	}

	for(i=0;i<tam*tam;i++)
	{
		matB[i] = 0.01f;//((float)i+4)/3;
	}
	
	matriz = MultiplicaMatriz(matA, matB, tam, rep);
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
