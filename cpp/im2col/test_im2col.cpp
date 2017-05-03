#include <stdlib.h>
#include <iostream>

#include "im2col.hpp"


int main()
{
    std::cout << "test_im2col: " << std::endl;

    int channels = 3;
    int height = 3;
    int width = 3;
    int kernel_h = 2;
    int kernel_w = 2;
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int size_im = channels * height * width;
    float * data_im;
    data_im = (float*)malloc(size_im * sizeof(data_im));
    for (int i = 0; i < size_im; ++i)
        data_im[i] = i;


    int kernel_dim = channels * kernel_h * kernel_w;
    int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    int output_h = (height + 2 * pad_h - kernel_extent_h) / stride_h + 1;
    int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    int output_w = (width + 2 * pad_w - kernel_extent_w) / stride_w + 1;
    int output_shape = output_h * output_w; // compute_output_shape()
    int size_col = kernel_dim * output_shape; 
    float * data_col;
    data_col = (float*)malloc(size_col * sizeof(data_col));

    im2col_cpu(data_im, channels, height, width, 
                        kernel_h, kernel_w,
                        pad_h, pad_w,
                        stride_h, stride_w,
                        dilation_h, dilation_w,
               data_col);

    for (int n = 0; n < channels; ++n)
    {
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << data_im[n*height*width + i*height + j] << ", ";
        }
        std::cout << std::endl;
    }
    }


    std::cout << "output_h: " << output_h << std::endl;
    std::cout << "output_w: " << output_w << std::endl;
    std::cout << "output_shape: " << output_shape << std::endl;
    for (int i = 0; i < kernel_dim; ++i)
    {
        for (int j = 0; j < output_shape; ++j)
        {
            std::cout << data_col[i*output_shape + j] << ", ";
        }
        std::cout << std::endl;
    }

    free(data_im);
    free(data_col);

    return 0;
}
