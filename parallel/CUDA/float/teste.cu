#define ERR_N_ARGS					1
#define OK							0
#define FAIL						2

#include "matrixMult.cuh"

int main(int argc, char **argv){
	int width;
	int rep;

	if(argc==1){
		width = 1024;
		rep = 1;
	}
	else if(argc == 3){
		width = atoi(argv[1]);
		rep = atoi(argv[2]);
	}
	else{
		printf("Modo de uso: %s <largura-da-matriz> <numero-repetições>\n",argv[0]);
		printf("Caso nenhum argumento seja dado, utiliza-se o tamanho 1024 com uma repetição.\n");
		printf("OBS.: Favor ajustar tamanho do ladrilho no arquivo matrixMult.cu\n");
		return ERR_N_ARGS;
	}

	float *res = matrixMultDevice(width,rep);
	int i;

	int tam = width*width;
	
	if(res!=NULL){
		if(DEBUG){
			printf("\n\n");
			for(i=0;i<tam;i++){
				if((i%width)==0 && i>0)
					printf("\n\t%.2f",res[i]);
				else
					printf("\t%.2f",res[i]);
			}
			printf("\n");
		}

		free(res);
		return OK;
	}
	else
		printf("Multiplicação falou.\n");
	
	return FAIL;
}
