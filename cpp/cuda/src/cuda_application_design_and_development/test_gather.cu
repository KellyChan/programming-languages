#include <iostream>

#include <omp.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;


struct gather_functor
{
    const int* index;
    const int* data;

    gather_functor(int * _data, int * _index) : data(_data), index(_index) {};
    __host__ __device__ int operator()(int i)
    {
        return data[index[i]];
    }
};


int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cerr << "Use: size (k) nLoops sequential" << endl;
        return (1);
    }

    int n = atof(argv[1]) * 1e3;
    int nLoops = atof(argv[2]);
    int op = atoi(argv[3]);
    cout << "Using " << (n/1.e6) << " M elements and averageing over " << nLoops << " tests" << endl;

    thrust::device_vector<int> d_a(n), d_b(n), d_index(n);
    thrust::sequence(d_a.begin(), d_a.end());
    thrust::fill(d_b.begin(), d_b.end(), -1);
    thrust::host_vector<int> h_index(n);

    switch(op)
    {
        case 0:
            // best case: sequential indicies
            thrust::sequence(d_index.begin(), d_index.end());
            cout << "Sequential data " << endl;
            break;
 
        case 1:
            // mid-performance case: randomo indices
            for (int i = 0; i < n; i++)
            {
                h_index[i] = rand() % (n-1);
                d_index = h_index;  // transfer to device
                thrust::sort(d_index.begin(), d_index.end());
                break;
            }

        default:
            // worse case: random indices
            for (int i = 0; i < n; i++)
            {
                h_index[i] = rand() % (n-1);
                d_index = h_index;  // transfer to device
                cout << "Random data " << endl;
                break;
            }
    }

    double startTime = omp_get_wtime();
    for (int i = 0; i < nLoops; i++)
    {
        thrust::transform(thrust::counting_iterator<unsigned int>(0),
                          thrust::counting_iterator<unsigned int>(n),
                          d_b.begin(),
                          gather_functor(thrust::raw_pointer_cast(&d_a[0]),
                                         thrust::raw_pointer_cast(&d_index[0])));

    }

    cudaDeviceSynchronize();
    
    double endTime = omp_get_wtime();

    // double check the results
    thrust::host_vector<int> h_b = d_b;
    thrust::host_vector<int> h_a = d_a;
    h_index = d_index;
    for (int i = 0; i < n; i++)
    {
        if (h_b[i] != h_a[h_index[i]])
        {
            cout << "Error!" << endl;
            return (1);
        }
    }

    cout << "Success!" << endl;
    cout << "Average time " << (endTime - startTime) / nLoops << endl;
}
