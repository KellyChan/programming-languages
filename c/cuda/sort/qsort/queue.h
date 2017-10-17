#include <stdio.h>
#include <cassert>

struct params {
    params() : output(0), input(0), len(0) {};
    int* output;
    int* input;    
    int len;    
    int left;
    int right;
};

#define MAX_QUEUE_SIZE 500
struct queue {
    queue()  {
        front = 0;
        back = 0;
        size = 0;
        data = (params*)malloc(sizeof(params)*MAX_QUEUE_SIZE);
    };
    ~queue() {
        free(data);
    }

    params* data    ;
    int front;
    int back;
    int size;
};

void enqueue(queue& q, params p) {
    if(q.size >= MAX_QUEUE_SIZE-1) {
        return;
    }
    
    q.data[q.back] = p;
    q.back++;
    q.size++;

    if(q.back >= MAX_QUEUE_SIZE) {
        q.back = 0;
    }
    //printf("(e)front:%d back:%d size:%d\n", q.front, q.back, q.size);
}

void dequeue(queue& q, params& p) {
    if(q.size <= 0) {
        return;
    }

    p = q.data[q.front];
    //printf("len%d\n", q.data[q.front].len);
    q.front++; 
    q.size--;   

    if(q.front >= MAX_QUEUE_SIZE) {
        q.front = 0;
    }    
    //printf("(d)front:%d back:%d size:%d\n", q.front, q.back, q.size);
}