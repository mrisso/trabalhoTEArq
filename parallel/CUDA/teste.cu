#include "matrixMult.cu"

int main(int argc, char **argv){
	float *res = matrixMultDevice(16);
	int i;

	int tam = 16*16;
	
	if(DEBUG){
		printf("\n\n");
		for(i=0;i<tam;i++){
			if((i%16)==0 && i>0)
				printf("\n\t%.2f",res[i]);
			else
				printf("\t%.2f",res[i]);
		}
		printf("\n");
	}

	free(res);
	return 0;
}
