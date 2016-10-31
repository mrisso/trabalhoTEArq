#include <stdio.h>
#include <stdlib.h>

float * MultiplicaMatriz(float * matA, float * matB, int tam, int rep)
{
	int i,j,k,lugar;
	float* matriz = (float*)malloc((tam*tam)*(sizeof(float)));
	float cValue = 0;
	
	/* matriz é a matriz resultado que queremos que seja a multiplicação de
	 * matA por matB nesta ordem.
	*/

	lugar = 0;
	
	for(i=0;i<rep;i++)
		for(j=0;j<tam;j++)
		{
			for(k=0;k<tam;k++)
			{
				cValue+=matA[j*tam+k] * matB[k*tam+j];
			}
			matriz[lugar]=cValue;
			lugar++;
			cValue = 0;
		}
		
	for(i=0;i<tam*tam;i++)
	{
		if(((i % (tam)) ==0) && i>0)
			printf("\n\t%.2f", matriz[i]);
		else
			printf("\t%.2f", matriz[i]);
	}
	printf("\n");
	printf("\n");
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
		matA[i] = 1;//((float)i+5)/7;
	}
	i=0;

	
	for(i=0;i<tam*tam;i++)
	{
		matB[i] = 2;//((float)i+4)/3;
	}
	printf("\n");
	
	matriz = MultiplicaMatriz(matA, matB, tam, rep);
	
	for(i=0;i<tam*tam;i++)
	{
		if(((i % (tam)) ==0) && i>0)
			printf("\n\t%.2f", matB[i]);
		else
			printf("\t%.2f", matB[i]);
	}
	printf("\n");
	printf("\n");
	printf("\n");
	for(i=0;i<tam*tam;i++)
	{
		if(((i % (tam)) ==0) && i>0)
			printf("\n\t%.2f", matA[i]);
		else
			printf("\t%.2f", matA[i]);
	}
	printf("\n");
	printf("\n");
	printf("\n");
	for(i=0;i<tam*tam;i++)
	{
		if(((i % (tam)) ==0) && i>0)
			printf("\n\t%.2f", matriz[i]);
		else
			printf("\t%.2f", matriz[i]);
	}
	printf("\n");
	printf("\n");
	printf("\n");
	return 0;
}
