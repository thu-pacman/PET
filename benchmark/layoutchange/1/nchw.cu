#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
using namespace std;

const int BlockDim = 32;
const int ThreadDim = 32;

const int batch = 128;
const int in_channel = 32;
const int in_size = 128;

void gen_tensor(float *a, int size) {
    size /= sizeof(float);
    for (int i = 0; i < size; ++i)
        a[i] = 1.0;
}

extern "C" __global__ void default_function_kernel0(void* __restrict__ A_change, void* __restrict__ A) {
  for (int n_inner = 0; n_inner < 4; ++n_inner) {
      for (int h_inner = 0; h_inner < 4; ++h_inner) {
            ((float4*)((float*)A_change + (((((((((int)blockIdx.y) * 2097152) + (n_inner * 524288)) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 512)) + (h_inner * 128)) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)((float*)A + (((((((((int)blockIdx.y) * 2097152) + (n_inner * 524288)) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 512)) + (h_inner * 128)) + (((int)threadIdx.x) * 4)))))[0];
      }
  }
}

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );    
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

double get_durtime(struct timeval t1, struct timeval t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

int main() {
    float *a, *d_a, *d_ach;
    int size = batch * in_channel * in_size * in_size * sizeof(float);
    
    a = (float *)malloc(size); gen_tensor(a, size);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_ach, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    dim3 grid(BlockDim, BlockDim);
    dim3 block(ThreadDim, ThreadDim);
    
    struct timeval t1, t2;

    gettimeofday(&t1, 0);
    default_function_kernel0 <<<grid, block>>> ((void *)d_ach, (void *)d_a);
    HANDLE_ERROR(cudaDeviceSynchronize());
    gettimeofday(&t2, 0);
    double conv_time = get_durtime(t1, t2);
    printf ("Convolution time: %f ms\n", conv_time);

    free(a);
    cudaFree(d_a), cudaFree(d_ach);
    return 0;
}

