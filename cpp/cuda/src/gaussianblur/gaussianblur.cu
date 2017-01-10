__global__ void gaussianBlurKernel (const unsigned char* const inputChannel,
                                    unsigned char* const outputChannel,
                                    int numRows,
                                    int numCols,
                                    const float* const filter,
                                    const int filterWidth)
{
    const int2 thread_2D_pos = make_int2 (blockDim.x * blockIdx.x + threadIdx.x,
                                          blockDim.y * blockIdx.y + threadIDx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return ;

    __syncthreads();

    float result = 0.0f;
    for (int x = -filterWidth/2; x <= filterWidth/2; x++)
    {
        for (int y = -filterWidth/2; y <= filterWidth/2; y++)
        {
            int valueX = min(max(thread_2D_pos.x + x, 0), numCols - 1);
            int valueY = min(max(thread_2D_pos.y + y, 0), numRows - 1);

            float channelValue = static_cast<float>(inputChannel[valueY * numCols + valueX]);
            float filterCoefficient = filter[x + filterWidth/2 + filterWidth * (y + filterWidth/2)];
            result += channelValue * filterCoefficient;
        }
    }

    outputChannel[thread_1D_pos] = result;
}


int main ()
{
    return 0;
} 
