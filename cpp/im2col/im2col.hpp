#include <iostream>

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}


/*
   im2col_cpu
*/
template <typename Dtype>
void im2col_cpu (const Dtype* data_im, 
                 const int channels,
                 const int height, 
                 const int width, 
                 const int kernel_h, 
                 const int kernel_w,
                 const int pad_h, 
                 const int pad_w,
                 const int stride_h, 
                 const int stride_w,
                 const int dilation_h, 
                 const int dilation_w,
                 Dtype* data_col) 
{
    std::cout << "(util) im2col_cpu: " << std::endl;
    std::cout << "(util::im2col_cpu) channels: " << channels << std::endl;
    std::cout << "(util::im2col_cpu) height: " << height << std::endl;
    std::cout << "(util::im2col_cpu) width: " << width << std::endl;
    std::cout << "(util::im2col_cpu) kernel_h: " << kernel_h << std::endl;
    std::cout << "(util::im2col_cpu) kernel_w: " << kernel_w << std::endl;
    std::cout << "(util::im2col_cpu) pad_h: " << pad_h << std::endl;
    std::cout << "(util::im2col_cpu) pad_w: " << pad_w << std::endl;
    std::cout << "(util::im2col_cpu) stride_h: " << stride_h << std::endl;
    std::cout << "(util::im2col_cpu) stride_w: " << stride_w << std::endl;
    std::cout << "(util::im2col_cpu) dilation_h: " << dilation_h << std::endl;
    std::cout << "(util::im2col_cpu) dilation_w: " << dilation_w << std::endl;
    

    // std::cout << "(util::im2col_cpu) data_im:" << std::endl;
    // for (int i = 0; i < channels; ++i)
    // {
    //     for (int j = 0; j < height; ++j)
    //     {
    //        for (int k = 0; k < width; ++k)
    //        {
    //            std::cout << data_im[i*height*width + j*width + k] << ", ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << std::endl;
    // }

    // output_h & output_w
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    std::cout << "(util::im2col_cpu) output_h = " << output_h << std::endl;
    std::cout << "(util::im2col_cpu) output_w = " << output_w << std::endl;

    // channel_size
    const int channel_size = height * width;
    std::cout << "(util::im2col_cpu) channel_size: " << channel_size << std::endl;
    std::cout << std::endl;

    for (int channel = channels; channel--; data_im += channel_size) 
    {
      std::cout << "(util::im2col_cpu) channel: " << channel << std::endl;
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) 
      {
        std::cout << "(util::im2col_cpu) kernel_row: " << kernel_row << std::endl;
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) 
      {
        std::cout << "(util::im2col_cpu) kernel_col: " << kernel_col << std::endl;
        int input_row = -pad_h + kernel_row * dilation_h;
        std::cout << "(util::im2col_cpu) input_row: " << input_row << std::endl;

        for (int output_rows = output_h; output_rows; output_rows--) 
        {
          std::cout << "(util::im2col_cpu) output_rows: " << output_rows << std::endl;
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) 
          {
            for (int output_cols = output_w; output_cols; output_cols--) 
            {
              *(data_col++) = 0;
            }
          } 
          else 
          {
            int input_col = -pad_w + kernel_col * dilation_w;
            std::cout << "(util::im2col_cpu) input_col: " << input_col << std::endl;

            for (int output_col = output_w; output_col; output_col--) 
            {
              std::cout << "(util::im2col_cpu) output_col: " << output_col << std::endl;
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) 
              {
                std::cout << "(util::im2col_cpu) im_index: " << input_row * width + input_col << std::endl;
                std::cout << "(util::im2col_cpu) data_im[x]: " << data_im[input_row * width + input_col] << std::endl;
                *(data_col++) = data_im[input_row * width + input_col];
              } 
              else 
              {
                *(data_col++) = 0;
              }
              input_col += stride_w;
              std::cout << "(util::im2col_cpu) input_col: " << input_col << std::endl;
              std::cout << std::endl;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }


    // std::cout << "(util::im2col_cpu) data_col:" << std::endl;
    // for (int i = 0; i < channels; ++i)
    // {
    //     for (int j = 0; j < height; ++j)
    //    {
    //        for (int k = 0; k < width; ++k)
    //        {
    //            std::cout << data_col[i*height*width + j*width + k] << ", ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << std::endl;
    // }
}
