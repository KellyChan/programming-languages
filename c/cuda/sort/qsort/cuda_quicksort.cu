#include <stdio.h>
#include <string>
#include <cassert>
#include <sys/time.h>
#include "queue.h"
#include <vector>
#include <fstream>
#include <string>
using namespace std;

#define ELEMENTS_PER_BLOCK 4096
#define BASE 10
//#define ELEMENTS_PER_BLOCK 512
#define THREADS_PER_BLOCK 512



#define CUDA_ASSERT(ans) gpuAssert((ans), __FILE__, __LINE__)
#define CUDA_KERNEL_ASSERT() CUDA_ASSERT(cudaPeekAtLastError()) || CUDA_ASSERT(cudaDeviceSynchronize()) 

inline  bool gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       // if (abort) exit(code);
        return false;
    }
    return true;
}

double tpartition = 0, tcount = 0, tprefix = 0, trarrange = 0, tssort = 0, tqsort = 0;

inline int rand_val_between(int min, int max) {
    return min + (rand() % (max - min + 1));
}

void print_array(int* a, int len) {
    for(int i = 0; i < len; ++i) {
        printf("%d\n", a[i]);
    }
}

// Thanks wikipedia.
void radixsort(int *a, int n) {
  int i, b[ELEMENTS_PER_BLOCK], m = a[0], exp = 1;
 
    for (i = 1; i < n; i++) {
        if (a[i] > m)
            m = a[i];
    }
     
    while (m / exp > 0) {
        int bucket[BASE] = { 0 }; 
        for (i = 0; i < n; i++)
          bucket[(a[i] / exp) % BASE]++;

        for (i = 1; i < BASE; i++)
          bucket[i] += bucket[i - 1];

        for (i = n - 1; i >= 0; i--)
          b[--bucket[(a[i] / exp) % BASE]] = a[i];
    
        for (i = 0; i < n; i++)
          a[i] = b[i];

        exp *= BASE;
    }
}


bool read_values_from_file(string file, vector<int>& values) {    
    ifstream fin;

    fin.open(file.c_str());
    if(fin.fail()) {       
        return false;
    }

    while(!fin.eof()) {           
        string line;
        int val;
    
        fin >> line;    
        val = atoi(line.c_str());

        values.push_back(val);                        
    }

    return true;
}

void cuda_print_memory() {
    size_t free_byte;
    size_t total_byte;

    CUDA_ASSERT(cudaMemGetInfo( &free_byte, &total_byte )) ;

    double free_db = (double)free_byte/1048576.0;
    double total_db = (double)total_byte/1048576.0;
    double used_db = total_db - free_db ;
    printf("\nDevice Memory:\n\tTotal:%.2fMB\n\tFree:%.2fMB\n\tUsed:%.2fMB\n\n", total_db, free_db, used_db);
}

void cuda_get_memory(double& free_db, double& total_db, double& used_db) {
    size_t free_byte;
    size_t total_byte;

    CUDA_ASSERT(cudaMemGetInfo( &free_byte, &total_byte )) ;

    free_db = (double)free_byte/1048576.0;
    total_db = (double)total_byte/1048576.0;
    used_db = total_db - free_db ;    
}


double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (1000000.0*(tv.tv_sec) + tv.tv_usec)/1000000.0;
}

int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

// output array needs to be padded. only works with one block
__device__ void prefix_scan(int* input, int* output, int len) {
    int i, j, k, step, nactive, tmp;
    int mytid = threadIdx.x;
    int nthreads = blockDim.x;
    __shared__ int npadded;

    if(mytid == 0) {
        for(npadded = 1; npadded < len; npadded *= 2);        
    }  

    __syncthreads();    

    for(i = mytid; i < len; i += nthreads) {
        output[i] = input[i];
    }

    for(i = len + mytid; i < npadded; i += nthreads) {        
        output[i] = 0;
    }
    
    for(step = 1, nactive = (npadded >> 1); nactive > 0; nactive >>= 1) {
        __syncthreads();
        for(k = mytid; k < nactive; k += nthreads) {
            i = step * (2*k+1) - 1;
            j = i + step;

            output[j] += output[i];            
        }
        step <<= 1;
    }  

    if(mytid == 0) {   
        output[npadded-1] = 0;
    }  

    for(nactive = 1; nactive < npadded; nactive *= 2) {    
        step >>= 1;
        __syncthreads();

        for(k = mytid; k < nactive; k += nthreads) {
            i = step * (2*k+1) - 1;
            j = i + step;

            tmp = output[i];
            output[i] = output[j];
            output[j] += tmp;
        }
    }
    __syncthreads();    
}

// count occurrences less than and greater than pivot value and store in lsum and usum params
__device__ void count(int* data, int len, int& lsum, int& usum, int pivot_value) {    
    int threadId = threadIdx.x;
    int num_threads = blockDim.x;   

    for(int i = threadId; i < len; i += num_threads) {          
        if(data[i] <= pivot_value) {
            atomicAdd(&lsum, 1);            
        } else {            
            atomicAdd(&usum, 1);
        }
    }
    __syncthreads();
}

// rearrange the input data to the output array
__device__ void rearrange(int* data, int len, int* output, int lower_offset, int upper_offset, int split, int pivot_value) {
    int threadId = threadIdx.x;

    int num_threads = blockDim.x; 
    __shared__ int li;
    __shared__ int ui;
    li = 0, ui = 0;
    int idx;
    for(int i = threadId; i < len; i += num_threads) {          
        __syncthreads();
        if(data[i] <= pivot_value) {
            idx = atomicAdd(&li, 1);            
            idx += lower_offset;                                                             
            output[idx] = data[i];                            
        } else {
            idx = atomicAdd(&ui, 1);            
            idx += split + upper_offset;                                             
            output[idx] = data[i];           
        }           
    }
    __syncthreads();
}

// counts the number of elements > and < than the pivot
__global__ void count_kernel(int* data, int len, int* lsums, int* usums, int pivot_value) {
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;  
    int num_blocks = gridDim.x;
    int block_len = (int)ceil((float)len / num_blocks);
    int block_start = block_len * blockId;
    int block_end = block_len * (blockId + 1);
    if(block_end >= len) {
        block_end = len ;
        block_len = block_end - block_start;
    }
    __shared__ int usum;
    __shared__ int lsum;    	

    usum = 0, lsum = 0;
    
    count(data + block_start, block_len, lsum, usum, pivot_value);
    
    if(threadId == 0) {
        lsums[blockId] = lsum;
        usums[blockId] = usum;    
    }
}

// performs perfix scan. Right now has to use 1 block, otherwise boom.
__global__ void prefix_scan_kernel(int* data, int len, int* lprefix, int* uprefix, int* lsums, int* usums, int len2) {    
    prefix_scan(lsums, lprefix, len2); 
    __syncthreads();   
    prefix_scan(usums, uprefix, len2);    
}

// sends the correct pointers to the rearrnage function based on the block
__global__ void rearrange_kernel(int* data, int* output, int len, int* lprefix, int* uprefix, int* lsums, int* usums, int pivot_value, int* split) {
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;  
    int num_blocks = gridDim.x;
    int block_len = (int)ceil((float)len / num_blocks);
    int block_start = block_len * blockId;
    int block_end = block_len * (blockId + 1);
    if(block_end >= len) {
        block_end = len ;
        block_len = block_end - block_start;
    }
    *split = lprefix[num_blocks-1] + lsums[num_blocks - 1];
    if(threadId == 0) {
        //printf("bid:%5d loffset:%14d uoffset:%14d lsum:%7d usum:%7d split:%d pivot:%14d\n", blockId, lprefix[blockId], uprefix[blockId], lsums[blockId], usums[blockId], *split, pivot_value);
        
    }
    rearrange(data + block_start, block_len, output, lprefix[blockId], uprefix[blockId], *split, pivot_value);        
}



void _cuda_partition(int* d_input, int* d_output, int len, 
                    int* d_lsums, int* d_usums, int* d_lprefix, int* d_uprefix, 
                    int* d_split, int pivot_value) {
    
    int num_blocks = ceil((double)len / ELEMENTS_PER_BLOCK); 
    double t1,t2;

    t1 = get_time();        
    count_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, len, d_lsums, d_usums, pivot_value);
    CUDA_KERNEL_ASSERT();
    t2 = get_time();   
    tcount += t2 - t1;     

    t1 = get_time();        
    prefix_scan_kernel<<<1, THREADS_PER_BLOCK>>>(d_input, len, d_lprefix, d_uprefix, d_lsums, d_usums, num_blocks);
    CUDA_KERNEL_ASSERT();    
    t2 = get_time();
    tprefix += t2 - t1;    

    t1 = get_time();        
    rearrange_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, len, d_lprefix, d_uprefix, d_lsums, d_usums, pivot_value, d_split);
    CUDA_KERNEL_ASSERT();    
    t2 = get_time();
    trarrange += t2 - t1;    
}



// using data parallelism and doing it breadth first
// doesnt work. cant figure out why, already spent too much time on it
void _cuda_quicksort_data(int* h_data,  int* d_input, int* d_output, int len, 
                     int* d_lsums, int* d_usums, int* d_lprefix, int* d_uprefix, 
                     int* d_split) {
    
    int split, pivot_index, pivot_value, m;
    queue q;
    params a;    
    a.input = d_input;
    a.output = d_output;
    a.len = len;
    a.left = 0;
    a.right = len;

    enqueue(q, a);

    while(q.size != 0) {
        params p;
        dequeue(q, p);
        if(p.len == 0) continue;

        if(p.len <= ELEMENTS_PER_BLOCK) {            
            int* temp = h_data + p.left;

            cudaMemcpy(temp, p.input, p.len*sizeof(int), cudaMemcpyDeviceToHost);          
            radixsort(temp, p.len);         

            // printf("sorting:%d right:%d len:%d\n", p.left, p.right, p.len);    
            // printf("%3d: %10d %s\n", p.left, temp[0], "yes");  
            // for(int i = 1; i < p.len; ++i) { 
            //     printf("%3d: %10d %s\n", p.left+i, temp[i], temp[i] >= temp[i-1] ? "yes" : "no");   
            // }    

 
        } else {
            pivot_index = rand_val_between(0, p.len-1);  
            pivot_value = 0;
            
            cudaMemcpy(&pivot_value, p.input + pivot_index, sizeof(int), cudaMemcpyDeviceToHost);
            _cuda_partition(p.input, p.output, p.len, d_lsums, d_usums, d_lprefix, d_uprefix, d_split, pivot_value);            

            CUDA_ASSERT(cudaMemcpy(&split, d_split, sizeof(int), cudaMemcpyDeviceToHost));        
            m = split + p.left;

            printf("left:%d split:%d right:%d len:%d\n", p.left, m, p.right, p.len);      

            // int* temp = (int*)malloc(sizeof(int)*p.len);
            // int* temp2 = (int*)malloc(sizeof(int)*p.len);
            // cudaMemcpy(temp, p.output + p.left, p.len*sizeof(int), cudaMemcpyDeviceToHost);  
            // cudaMemcpy(temp2, p.input + p.left, p.len*sizeof(int), cudaMemcpyDeviceToHost);  
            // for(int i = 0; i < p.len; ++i) { 
            //     printf("%3d: %10d %10d %2s %d\n", p.left+i, temp2[i], temp[i], temp[i] < pivot_value ? "<  " : (temp[i] > pivot_value ? "  >" : " = "), pivot_value);
            //      if(i < split) {
            //          assert(temp[i] <= pivot_value);                                             
            //      else
            //         assert(temp[i] > pivot_value);                   
            // }
            // free(temp);
            // free(temp2);

            params l; 
            l.input = p.output; 
            l.output = p.input;                
            l.len = split;
            l.left = p.left;
            l.right = m;            
            enqueue(q, l);

            params r;          
            r.input = p.output + split;
            r.output = p.input + split;                    
            r.len = p.len - split;
            r.left = m;
            r.right = p.right;
            enqueue(q, r);              
        }                    
    }
}

// data parallelism but using recursion, so depth first instead of breadth first
void _cuda_quicksort_recur(int* h_data,  int* d_input, int* d_output, int len, 
                           int* d_lsums, int* d_usums, int* d_lprefix, int* d_uprefix, 
                           int* d_split, int left, int right, int depth = 0) {
    double t1,t2;
    int pivot_index = rand_val_between(0, len-1);  
    int pivot_value = 0;
    cudaMemcpy(&pivot_value, d_input + pivot_index, sizeof(int), cudaMemcpyDeviceToHost);
    
    int split;    

    if(len <= ELEMENTS_PER_BLOCK) {        
        int* temp = h_data+left;
        t1 = get_time();        
        
        cudaMemcpy(temp, d_input, len*sizeof(int), cudaMemcpyDeviceToHost);
        radixsort(temp, len);         

        t2 = get_time();
        tssort += t2 - t1;
        
        

        // printf("%3d: %10d %s\n", left, temp[0], "yes");  
        // for(int i = 1; i < len; ++i) { 
        //     printf("%3d: %10d %s\n", i+left, temp[i], temp[i] >= temp[i-1] ? "yes" : "no");   
        // }    
        return;
    } 
    
    t1 = get_time();        
    
    _cuda_partition(d_input, d_output, len, d_lsums, d_usums, d_lprefix, d_uprefix, d_split, pivot_value);
    CUDA_ASSERT(cudaMemcpy(&split, d_split, sizeof(int), cudaMemcpyDeviceToHost));        
    
    t2 = get_time();
    tpartition += t2 - t1;    

    // check partition is correct
    // int* temp = (int*)malloc(sizeof(int)*len);
    // int* temp2 = (int*)malloc(sizeof(int)*len);
    // cudaMemcpy(temp, d_output, len*sizeof(int), cudaMemcpyDeviceToHost);  
    // cudaMemcpy(temp2, d_input, len*sizeof(int), cudaMemcpyDeviceToHost);  
    // for(int i = 0; i < len; ++i) { 
    //     printf("%3d: %10d %10d %2s %d\n", i+left, temp2[i], temp[i], temp[i] < pivot_value ? "<  " : (temp[i] > pivot_value ? "  >" : " = "), pivot_value);
    //      if(i < split) {
    //        assert(temp[i] <= pivot_value);
    //      } else {
    //        assert(temp[i] > pivot_value);
    //      }    
    // }
    // free(temp);
    // free(temp2);
    
    // printf("%d l:%d m:%d(%d) r:%d\n", depth, left, left+split, split, right);
    if(split != 0) {                                
        // printf("%d (l) s:%d e:%d len:%d\n", depth, left, left+split, split); 
        _cuda_quicksort_recur(h_data, d_output, d_input, split, 
                      d_lsums, d_usums, d_lprefix, d_uprefix, 
                      d_split, left, left+split, depth + 1);
    }

    if(split != len) {                        
        // printf("%d (r) s:%d e:%d len:%d\n", depth, left+split, left+len, len - split);
        _cuda_quicksort_recur(h_data, d_output + split, d_input + split, len - split, 
                        d_lsums, d_usums, d_lprefix, d_uprefix, d_split, 
                        left+split, len, depth + 1);
    }
} 

void cuda_qsort(int* data, int len) {        
    int* d_input, *d_lsums, *d_usums, *d_lprefix,*d_uprefix, *d_output;    
    int* d_split;

    int size = len * sizeof(int);   
    int num_blocks = ceil((double)len / ELEMENTS_PER_BLOCK); 
    int sum_array_size = sizeof(int) * num_blocks * 2;
    int prefix_array_size = sum_array_size * 2; //needs padding    
            
    CUDA_ASSERT(cudaMalloc((void**)&d_split, sizeof(int)));      
    CUDA_ASSERT(cudaMalloc((void**)&d_input, size));      
    CUDA_ASSERT(cudaMalloc((void**)&d_output, size));          
    CUDA_ASSERT(cudaMalloc((void**)&d_lprefix, prefix_array_size));      
    CUDA_ASSERT(cudaMalloc((void**)&d_uprefix, prefix_array_size));   
    CUDA_ASSERT(cudaMalloc((void**)&d_lsums, sum_array_size));      
    CUDA_ASSERT(cudaMalloc((void**)&d_usums, sum_array_size));          
    CUDA_ASSERT(cudaMemcpy(d_input, data, size, cudaMemcpyHostToDevice));      

    double t1 = get_time();
    _cuda_quicksort_recur(data, d_input, d_output, len, d_lsums, d_usums, d_lprefix, d_uprefix, d_split, 0, len);
    double t2 = get_time();
    tqsort = t2 - t1;    
    
    // cleanup    
    CUDA_ASSERT(cudaFree(d_input));        
    CUDA_ASSERT(cudaFree(d_output));  
    CUDA_ASSERT(cudaFree(d_uprefix));     
    CUDA_ASSERT(cudaFree(d_lprefix));    
    CUDA_ASSERT(cudaFree(d_usums));     
    CUDA_ASSERT(cudaFree(d_lsums));         
}


int main(int argc, char** argv) {
    int* data = 0;
    int* data2 = 0;
    int len;
    double t1, t2;

    if(argc != 2) {
        printf("usage - cuda_quicksort <filename>\n");
    }
    char* file = argv[1];
    vector<int> temp;
    if(!read_values_from_file(file, temp)) {
        printf("failed to read values from file\n");
        return -1;
    }
    data = temp.data();
    len = temp.size();

    srand(time(NULL));      

    

    data2 = (int*)malloc(len * sizeof(int));    
    memcpy(data2, data, len*sizeof(int));

    t1 = get_time();
    qsort(data2, len, sizeof(int), cmpfunc);
    t2 = get_time();
    printf("libc qsort: %fs\n", t2 - t1);
    
   // cuda_qsort(data, len);    
    printf("cuda_qsort: %fs (%fs partitioning & %fs host sort)\n", tqsort, tpartition, tssort);
    
    for(int i = 1; i < len; ++i) {  
        if(data[i] < data[i-1])  {
          printf("\n\n\nUnsorted: Value at index %d is %d. Value at index %d is %d\n", i - 1, data[i - 1], i, data[i]);    
        }
        assert(data[i] >= data[i-1]);        
    }
    printf("Sorted Values:\n");
    print_array(data, len);

    // clean up    
    free(data2);
    return 0;
}

