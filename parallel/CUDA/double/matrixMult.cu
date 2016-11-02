#define TILE_WIDTH				32 //Tamanho do ladrilho é definido aqui para a utilização de memória compartilhada

#include "matrixMult.cuh"

//kernel
__global__ void matrixMultKernel(double *dC, double *dA, double *dB, long width){

	//Utilizar memória compartilhada para o ladrilho
	__shared__ double dAs[TILE_WIDTH][TILE_WIDTH];
	__shared__ double dBs[TILE_WIDTH][TILE_WIDTH];

	long bx = blockIdx.x; long by = blockIdx.y;
	long tx = threadIdx.x; long ty = threadIdx.y;

	//Cálculo de linha e coluna
	long lin = by * TILE_WIDTH + ty;
	long col = bx * TILE_WIDTH + tx;
	
	double cValue = 0; //Acumulador

	//Cálculo
	__syncthreads();
	for(long k = 0;k < width/TILE_WIDTH ; k++){
		dAs[ty][tx] = dA[lin * width + (k * TILE_WIDTH + tx)];
		dBs[ty][tx] = dB[(k * TILE_WIDTH + ty) * width + col];
		__syncthreads();

		for(long i = 0; i<TILE_WIDTH; i++)
			cValue += dAs[ty][k] * dBs[k][tx];
		__syncthreads();
	}

	dC[lin * width + col] = cValue; //Resultado
}

//Auxiliar para popular matrizes
void popularMatriz(double *matriz, int tam, double valor){
	for(int i = 0; i < tam; i++){
		matriz[i] = valor;
	}
}
double *matrixMultDevice(int width, int rep){
	int size = width*width;
	int i;
	int sizeM = size*sizeof(double);

	//Arranjar matrizes no host
	double *hA = (double*) malloc(sizeM);
	if(hA==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	double *hB = (double*) malloc(sizeM);
	if(hB==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	double *hC = (double*) malloc(sizeM);
	if(hC==NULL){
		printf("Malloc error on host!\n");
		exit(MALLOC_ERROR);
	}

	//--> Populando matrizes A e B
	popularMatriz(hA,size,1.0f);
	popularMatriz(hB,size,0.01f);

	if(DEBUG){
		for(i=0;i<size;i++){
			if((i%width)==0 && i>0)
				printf("\n\t%.2f",hA[i]);
			else
				printf("\t%.2f",hA[i]);
		}

		printf("\n\n");

		for(i=0;i<size;i++){
			if((i%width)==0 && i>0)
				printf("\n\t%.2f",hB[i]);
			else
				printf("\t%.2f",hB[i]);
		}
		printf("\n\n");
	}

	//Alocar matrizes no device
	double *dA,*dB,*dC;

	cudaError_t error;

	error = cudaMalloc((void **) &dA,sizeM);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc((void **)&dB,sizeM);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMalloc((void **)&dC,sizeM);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

	//Copying memory to device
	error = cudaMemcpy(dA,hA,sizeM,cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

	error = cudaMemcpy(dB,hB,sizeM,cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	//Configurar execução
	dim3 dimGrid(width/TILE_WIDTH,width/TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

	//Iniciar contagem de tempo
	clock_t t;
	t = clock();

	//Realizar multiplicação
	for(int i = 0; i<rep; i++)
		matrixMultKernel<<<dimGrid, dimBlock>>>(dC,dA,dB,width);

	//Terminar contagem de tempo
	t = clock() - t;
	double tempoExePar = ((double)t)/CLOCKS_PER_SEC;

	//Copiar resultado para o host
	error = cudaMemcpy(hC,dC,sizeM,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

	t = clock();

	//Executar codigo sequencial para comparação
	double *hCSeq;
	hCSeq = multiplicaMatriz(hA,hB,width,rep);

	t = clock() - t;
	double tempoExeSeq = ((double)t)/CLOCKS_PER_SEC;

	int res = compareRes(hC,hCSeq,size);

	printf("Tamanho: %d\n\n",width);
	printf("Multiplicação Paralela: %lfs\n",tempoExePar);
	printf("Multiplicação Sequencial: %lfs\n",tempoExeSeq);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	free(hA);
	free(hB);
	if(res)
		return hC;
	return NULL;
}

int compareNum(double n1, double n2, double precisao){
	if(n1==n2)
		return 1;
	if(n1>n2 && n1<=(n2-precisao))
		return 1;
	else if(n1<n2 && n1>=(n2-precisao))
		return 1;
	return 0;
}

int compareRes(double *mA, double *mB, int size){
	int correct = 1;
	for(int i = 0; i<size; i++){
		if(!compareNum(mA[i],mB[i],0.01f)){
			printf("Resultado errado para posição %d: %f!=%f\n",i,mA[i],mB[i]);
			correct = 0;
		}
	}
	return correct;
}
