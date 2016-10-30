#include "matrixMult.cuh"

//kernel
__global__ void matrixMultKernel(float *dC, float *dA, float *dB, int width){ 
	//Cálculo de linha e coluna
	int lin = blockIdx.y*blockDim.y+ threadIdx.y; int col = blockIdx.x*blockDim.x+ threadIdx.x;

	float cValue = 0; //Acumulador

	//Cálculo para cada thread (um elemento do ladrilho)
	for(int k = 0; k < width; k++){
		cValue += dA[lin*width + k] * dB[k*width + col];
	}

	dC[lin*width + col] = cValue; //Resultado
}

//Auxiliar para popular matrizes
void popularMatriz(float *matriz, int tam, float valor){
	for(int i = 0; i < tam; i++){
		matriz[i] = valor;
	}
}

float *matrixMultDevice(int width, int tileW){
	int size = width*width;

	//Arranjar matrizes no host
	float *hA = (float*) malloc(size*sizeof(float));
	if(hA==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	float *hB = (float*) malloc(size*sizeof(float));
	if(hB==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	float *hC = (float*) malloc(size*sizeof(float));
	if(hC==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	//--> Populando matrizes A e B
	popularMatriz(hA,size,1.0);
	popularMatriz(hB,size,1.0);

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

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	free(hA);
	free(hB);
	return hC;
}

int compareNum(float n1, float n2, float precisao){
	if(n1==n2)
		return 1;
	if(n1>n2 && n1<=(n2-precisao))
		return 1;
	else if(n1<n2 && n1>=(n2-precisao))
		return 1;
	return 0;
}

void compareRes(float *mA, float *mB, int size){
	for(int i = 0; i<size; i++){
		if(!compareNum(mA[i],mB[i],1.0))
			printf("Resultado errado para posição %d: %f!=%f\n",i,mA[i],mB[i]);
	}
}
