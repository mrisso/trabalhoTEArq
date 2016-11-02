#Compilador
CC = gcc 

#Opções de compilação
CFLAGS = -Wall

#Arquivos de CF
FONTES = $(wildcard *.c)

OBJS = $(FONTES:.c=.o)

#Nome do executável
EXE = multiply

#all
all: $(EXE)

$(EXE): $(OBJS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) -c $(CFLAGS) $^

clean:
	rm -rf *.o

rmf: clean
	rm -rf $(EXE)
