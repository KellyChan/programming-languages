#include <stdio.h>
#include <cstdlib>
#include <iostream>

typedef struct
{
   int a;
   int b;
   float *c; 
} Dummy;


void printValues(Dummy *A, int c_memsize, int c_valsize)
{
    std::cout << A->a << std::endl;
    std::cout << A->b << std::endl;

    // A->c
    if (c_memsize >= c_valsize)
    {
        for (int i = 0; i < c_memsize; ++i)
            std::cout << A->c[i] << std::endl;
    }
    else
    {
        for (int i = 0; i < c_valsize; ++i)
            std::cout << A->c[i] << std::endl;
    }
}


int main ()
{

    int c_memsize = 20;
    int c_valsize = 20;

    // c
    float *c;
    c = (float*)malloc(c_memsize*sizeof(float));
    for (int i = 0; i < c_valsize; ++i)
        c[i] = i;

    // dummy
    Dummy *test = (Dummy*)malloc(sizeof(Dummy));
    test->a = 1;
    test->b = 2;
    test->c = c;
    printValues(test, c_memsize, c_valsize);

    // free
    free(test);
    free(c);

    return 0;
}
