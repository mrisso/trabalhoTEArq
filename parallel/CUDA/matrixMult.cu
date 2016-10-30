#define MALLOC_ERROR									1

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

//Kernel
_global_ void matrixMultKernel(float *dC, float *dA, float *dB, int width){
	
	//Cálculo de linha e coluna
	int lin = blockIdx.y*blockDim.y+ threadIdx.y;
	int col = blockIdx.x*blockDim.x+ threadIdx.x;

	float cValue = 0; //Acumulador

	//Cálculo para cada thread (um elemento do ladrilho)
	for(int k = 0; k < width; k++){
		cValue += dA[lin*width + k] * dB[k*width + col];
	}

	dC[lin*width + col] = cValue; //Resultado
}

//Auxiliar para popular matrizes
void popularMatriz(float *matriz, int tam, float valor){
	for(int i = 0; i < size; i++){
		data[i] = valor;
	}
}

void matrixMultDevice(int width, int tileW){
	int size = width*width;

	//Arranjar matrizes no host
	if((float *hA = (float*) malloc(size*sizeof(float)))==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	if((float *hB = (float*) malloc(size*sizeof(float)))==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	if((float *hC = (float*) malloc(size*sizeof(float)))==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	//--> Populando matrizes A e B
	popularMatriz(hA,size,1.0);
	popularMatriz(hB,size,0.1);

	//Alocar matrizes no device
	float *dA,*dB,*dC;

	cudaMalloc(&dA,size);
	cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);

	cudaMalloc(&dB,size);
	cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice);

	cudaMalloc(&dC,size);
	
	//Configurar execução
	dim3 dimGrid(width/tileW,width,tileW);
	dim3 dimBlock(tileW,tileW);

	//Realizar multiplicação
	matrixMultKernel<<<dimGrid, dimBlock>>>(dC,dA,dB,width);

	//Copiar resultado para o host
	cudaMemcpy(hC,dC,size,cudaMemcpyDeviceToHost);

}
