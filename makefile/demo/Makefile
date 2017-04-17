IDIR=../include
CC=gcc
CFLAGS=-i$(IDIR)

ODIR=obj
LDIR=../lib

LIBS=-lm

_DEPS=hellomake.h
DEPS=$(patsubst %, $(ODIR)/%, $(_OBJ))

_OBJ = hellomake.o hellofunc.o
OBJ=$(patsubst %, $(ODIR)/%, $(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
	gcc -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
