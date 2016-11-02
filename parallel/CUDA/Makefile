#Compilador
CC = nvcc

#Opções de compilação
CFLAGS = #-Wall

#Arquivos de CF
FONTES = $(wildcard *.cu)

OBJSCU = $(FONTES:.cu=.o)

OBJSC = seqMatrix.o

#Nome do executável
EXE = multiply

#all
all: $(EXE)

$(EXE): $(OBJSCU) $(OBJSC)
	$(CC) -o $@ $^

$(OBJSC): 
	$(CC) -c ../../sequential/float/seqMatrix.c

%.o: %.cu
	$(CC) -c $(CFLAGS) $^

clean:
	rm -rf *.o

rmf: clean
	rm -rf $(EXE)
