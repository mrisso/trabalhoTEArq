#include <stdio.h>
#include <stdlib.h>

float * MultiplicaMatriz(float * matA, float * matB, int tam, int rep)
{
	int i=0, j=0;
	float* matriz = (float*)malloc((tam*tam)*(sizeof(float)));
	
	/* matriz é a matriz resultado que queremos que seja a multiplicação de
	 * matA por matB nesta ordem.
	*/
	
	for(i=0; i<rep; i++)
	{
		if((i % 2)==0)
		{
			matriz = MultiplicaMatriz(matriz, matA,tam,0);
		}
		else
		{
			matriz = MultiplicaMatriz(matriz, matB,tam,0);
		}
	}
	
	return matriz;
}

int main (int argc, char *argv[])
{
	int i=0,j=0;
	int tam = atoi(argv[1]);
	int rep = atoi(argv[2]);
	
	float * matA = (float*)malloc(tam*tam*sizeof(float));
	float * matB = (float*)malloc(tam*tam*sizeof(float));
	float * matriz = (float*)malloc(tam*tam*sizeof(float));
	
	for(i=0;i<tam*tam;i++)
	{
		matA[i] = ((float)i+5)/7;
	}
	i=0;

	
	for(i=0;i<tam*tam;i++)
	{
		matB[i] = ((float)i+4)/3;
	}
	printf("\n");
	
	matriz = MultiplicaMatriz(matA, matB, tam, rep);
	

	return 0;
}
