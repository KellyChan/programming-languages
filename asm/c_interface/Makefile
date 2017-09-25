all:
	nasm -f elf -o add.o add.asm
	gcc -c main.c -o main.o
	gcc -o test_asm add.o main.o
