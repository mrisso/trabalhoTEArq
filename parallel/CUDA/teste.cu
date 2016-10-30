#include "matrixMult.cu"

int main(int argc, char **argv){
	float *res = matrixMultDevice(1024,16);

	int tam = 1024*1024;

	for(int i = 0;i<tam;i++)
		printf("%f\n",res[i]);

	free(res);
	return 0;
}
