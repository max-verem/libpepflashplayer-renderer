#include <stdint.h>
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "drvapi_error_string.h"

#include "../common/log.h"

__global__ void cu_interlace_frames(uint32_t* src_0, uint32_t* src_1, uint32_t* dst, int width, int height)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int s = i + j * width;
    int d = i + (height - 1 - j) * width;

    uint32_t src = (j & 1) ? src_1[s] : src_0[s];

    dst[d] = src;
};

extern "C" int cuda_interlace_frames(unsigned char* src_0, unsigned char* src_1, unsigned char* dst, int stride, int height, cudaStream_t cu_stream)
{
    int width = stride / sizeof(uint32_t);
    dim3 threads(16, 8, 1);
    dim3 blocks(width / threads.x, height / threads.y, 1);

    cu_interlace_frames<<<blocks, threads, 0, cu_stream>>>((uint32_t*)src_0, (uint32_t*)src_1, (uint32_t*)dst, width, height);

    return 0;
};
