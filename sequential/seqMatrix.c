#include "seqMatrix.h"

float * multiplicaMatriz(float * matA, float * matB, int tam, int rep) //Função recebe 2 matrizes (matA e matB), tamanho de uma dimensão da matriz quadrada (tam) e quantas repetições da mesma multiplicação serão feitas (rep).//
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
