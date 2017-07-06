#include <stdio.h>
#include <stdlib.h>

int *TheArray;

int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (TheArray[ia] > TheArray[ib]) - (TheArray[ia] < TheArray[ib]);
}

int main(void) {
    int a[] = {2,3,4,5,8,2,5,6};
    size_t len = sizeof(a)/sizeof(*a);
    int a_index[len];
    int i;

    for(i = 0; i < len; ++i)
        a_index[i] = i;

    TheArray = a;
    qsort(a_index, len, sizeof(*a_index), cmp);

    for(i = 0; i < len; ++i)
        printf("%d ", a_index[i]);//5 0 1 2 6 3 7 4 : qsort is not a stable.
    printf("\n");

    return 0;
}
