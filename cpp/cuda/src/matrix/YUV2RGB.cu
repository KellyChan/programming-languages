/* YUV to RGB Conversion

Ref: http://www.nvidia.com/content/nvision2008/tech_presentations/Game_Developer_Track/NVISION08-Image_Processing_and_Video_with_CUDA.pdf


CUDA Kernel performs YUV to RGB

    R   | 1.0     0      1.402     |  | Y  |
    G = | 1.0  -0.34413  -0.714136 |  | Cb |
    B   | 1.0  1.772     0         |  | Cr |

*/


__global__ void YUV2RGB(uint32* yuvi,
                        float*  R,
                        float*  G,
                        float*  B)
{
    float luma;
    float chromaCb;
    float chromaCr;

    // Prepare for hue adjustment (10-bit YUV to RGB)
    luma     = (float)yuvi[0];
    chromaCb = (float)((int32)yuvi[1] - 512.0f);
    chromaCr = (float)((int32)yuvi[2] - 512.0f);

    // Convert YUV to RGB with hue adjustment
    *R       = MUL(luma,     constHueColorSpaceMat[0]) +
               MUL(chromaCb, constHueColorSpaceMat[1]) +
               MUL(chromaCr, constHueColorSpaceMat[2]);
    *G       = MUL(luma,     constHueColorSpaceMat[3]) +
               MUL(chromaCb, constHueColorSpaceMat[4]) +
               MUL(chromaCr, constHueColorSpaceMat[5]);
    *B       = MUL(luma,     constHueColorSpaceMat[6]) +
               MUL(chromaCb, constHueColorSpaceMat[7]) +
               MUL(chromaCr, constHueColorSpaceMat[8]);
}
