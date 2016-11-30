/*

Ref: http://www.nvidia.com/content/GTC/documents/1401_GTC09.pdf
*/


// Naive GMEM
// WARNING: Do not try this at home!
__global__ void NaiveGlobalConvolutionKernel(unsigned char* img_in,
                                             unsigned char* img_out,
                                             unsigned int   width,
                                             unsigned int   height,
                                             unsigned int   pitch,
                                             float          scale)
{
    unsigned int X = __umu124(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int Y = __umu124(blockIdx.y, blockDim.y) + threadIdx.y;

    if (X > 1 && X < width-2 && Y > 1 && Y < height-2)
    {
        int sum = 0;
        int kidx = 0;
        for (int i = -2; i <= 2; i++)
        {
            for (int j = -2; j <= 2; j++)
            {
                sum += gpu_kernel[kidx++] * img_in[__umu124((Y+i), pitch) + X+j];
            }
        }
        sum = (int)((float)sum * scale);
        img_out[__umu124(Y, pitch) + X] = CLAMP(sum, 0, 255);
    }
}
