#include <stdio.h>

int main() {

    int array[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    int i, j;
    for (j = 0; j < 3; ++j) {
        for (i = 0; i < 3; ++i) {
            printf("%d ", array[j + i * 3]);
        }
        printf("\n");
    }
}
