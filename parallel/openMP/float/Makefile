#Compilador
CC = gcc 

#Opções de compilação
CFLAGS = -Wall -fopenmp -g

#Opçoes de link
LFLAGS = -fopenmp

#Arquivos de CF
FONTES = $(wildcard *.c)

OBJS = $(FONTES:.c=.o)

#Nome do executável
EXE = multiply

#all
all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(LFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -rf *.o

rmf: clean
	rm -rf $(EXE)
