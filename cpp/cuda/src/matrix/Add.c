#include <stdio.h>

# define N 10


void AddCPU(int* a, int* b, int* c)
{
    int tid = 0;
    while (tid < N)
        c[tid] = a[tid] + b[tid];
        tid += 1;
}


int addcpu(void)
{

    int a[N], b[N], c[N];

    // Fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    AddCPU(a, b, c);

    printf(" --- CPU --- \n");
    // Display the results
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}


int main(void)
{
    printf(" --- CPU --- \n");
    addcpu();

    return 0;
}
