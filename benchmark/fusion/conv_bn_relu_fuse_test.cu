#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );    
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void conv(void* __restrict__ A, void* __restrict__ W, void* __restrict__ B) {
  float B_local[64];
  __shared__ float Apad_shared[512];
  __shared__ float W_shared[512];
  float Apad_shared_local[8];
  float W_shared_local[8];
  for (int ff_c_init = 0; ff_c_init < 8; ++ff_c_init) {
    for (int nn_c_init = 0; nn_c_init < 8; ++nn_c_init) {
      B_local[(((ff_c_init * 8) + nn_c_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        __syncthreads();
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
          ((float4*)(Apad_shared + ((((((int)threadIdx.x) * 64) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer * 4)))))[0] = (((((1 <= ((((int)blockIdx.z) / 14) + ry)) && (((((int)blockIdx.z) / 14) + ry) < 15)) && (1 <= (rx + (((int)blockIdx.z) % 14)))) && ((rx + (((int)blockIdx.z) % 14)) < 15)) ? ((float4*)((float*)A + ((((((((((ry * 458752) + (((int)blockIdx.z) * 32768)) + (rx * 32768)) + (rc_outer * 1024)) + (((int)threadIdx.x) * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer * 4)) - 491520))))[0] : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
        }
        for (int ax3_inner_outer1 = 0; ax3_inner_outer1 < 2; ++ax3_inner_outer1) {
          ((float4*)(W_shared + ((((((int)threadIdx.x) * 64) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer1 * 4)))))[0] = ((float4*)((float*)W + ((((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + (((int)threadIdx.x) * 512)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer1 * 4)))))[0];
        }
        __syncthreads();
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          for (int ax3 = 0; ax3 < 8; ++ax3) {
            Apad_shared_local[(ax3)] = Apad_shared[((((rc_inner * 64) + (((int)threadIdx.x) * 8)) + ax3))];
          }
          for (int ax31 = 0; ax31 < 8; ++ax31) {
            W_shared_local[(ax31)] = W_shared[((((rc_inner * 64) + (((int)threadIdx.x) * 8)) + ax31))];
          }
          for (int ff_c = 0; ff_c < 8; ++ff_c) {
            for (int nn_c = 0; nn_c < 8; ++nn_c) {
              B_local[(((ff_c * 8) + nn_c))] = (B_local[(((ff_c * 8) + nn_c))] + ((Apad_shared_local[(nn_c)] * W_shared_local[(ff_c)])));
            }
          }
        }
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 8; ++ff_inner_inner_inner) {
    for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 8; ++nn_inner_inner_inner) {
      ((float*)B)[((((((((((int)blockIdx.z) * 65536) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.x) * 1024)) + (ff_inner_inner_inner * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + nn_inner_inner_inner))] = B_local[(((ff_inner_inner_inner * 8) + nn_inner_inner_inner))];
    }
  }
}

__global__ void conv_bn_relu(void* __restrict__ A, void* __restrict__ W, void* __restrict__ mean, void* __restrict__ scale, 
                             void* __restrict__ var, void* __restrict__ beta, void* __restrict__ B) {
  float B_local[64];
  float mean_local[8];
  float scale_local[8];
  float var_local[8];
  float beta_local[8];
  float epsilon = 1e-5;
  __shared__ float Apad_shared[512];
  __shared__ float W_shared[512];
  float Apad_shared_local[8];
  float W_shared_local[8];
  for (int ff_c = 0; ff_c < 8; ++ff_c) {
    mean_local[ff_c] = ((float*)mean)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ff_c))];
    scale_local[ff_c] = ((float*)scale)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ff_c))];
    var_local[ff_c] = ((float*)var)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ff_c))];
    beta_local[ff_c] = ((float*)beta)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ff_c))];
  }
  for (int ff_c_init = 0; ff_c_init < 8; ++ff_c_init) {
    for (int nn_c_init = 0; nn_c_init < 8; ++nn_c_init) {
      B_local[(((ff_c_init * 8) + nn_c_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        __syncthreads();
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
          ((float4*)(Apad_shared + ((((((int)threadIdx.x) * 64) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer * 4)))))[0] = (((((1 <= ((((int)blockIdx.z) / 14) + ry)) && (((((int)blockIdx.z) / 14) + ry) < 15)) && (1 <= (rx + (((int)blockIdx.z) % 14)))) && ((rx + (((int)blockIdx.z) % 14)) < 15)) ? ((float4*)((float*)A + ((((((((((ry * 458752) + (((int)blockIdx.z) * 32768)) + (rx * 32768)) + (rc_outer * 1024)) + (((int)threadIdx.x) * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer * 4)) - 491520))))[0] : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
        }
        for (int ax3_inner_outer1 = 0; ax3_inner_outer1 < 2; ++ax3_inner_outer1) {
          ((float4*)(W_shared + ((((((int)threadIdx.x) * 64) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer1 * 4)))))[0] = ((float4*)((float*)W + ((((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + (((int)threadIdx.x) * 512)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer1 * 4)))))[0];
        }
        __syncthreads();
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          for (int ax3 = 0; ax3 < 8; ++ax3) {
            Apad_shared_local[(ax3)] = Apad_shared[((((rc_inner * 64) + (((int)threadIdx.x) * 8)) + ax3))];
          }
          for (int ax31 = 0; ax31 < 8; ++ax31) {
            W_shared_local[(ax31)] = W_shared[((((rc_inner * 64) + (((int)threadIdx.x) * 8)) + ax31))];
          }
          for (int ff_c = 0; ff_c < 8; ++ff_c) {
            for (int nn_c = 0; nn_c < 8; ++nn_c) {
              B_local[(((ff_c * 8) + nn_c))] = (B_local[(((ff_c * 8) + nn_c))] + ((Apad_shared_local[(nn_c)] * W_shared_local[(ff_c)])));
            }
          }
        }
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 8; ++ff_inner_inner_inner) {
    for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 8; ++nn_inner_inner_inner) {
      ((float*)B)[((((((((((int)blockIdx.z) * 65536) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.x) * 1024)) + (ff_inner_inner_inner * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + nn_inner_inner_inner))] = max(0.0, scale_local[ff_inner_inner_inner] * ( B_local[(((ff_inner_inner_inner * 8) + nn_inner_inner_inner))] - mean_local[ff_inner_inner_inner] ) / sqrt( var_local[ff_inner_inner_inner] + epsilon ) + beta_local[ff_inner_inner_inner]);
    }
  }
}

//extern "C" 
__global__ void default_function_kernel0(void* __restrict__ A, void* __restrict__ W, void* __restrict__ beta, void* __restrict__ B) {
  float B_local[64];
  float beta_local[8];
  __shared__ float Apad_shared[512];
  __shared__ float W_shared[512];
  float Apad_shared_local[8];
  float W_shared_local[8];
  for (int ff_c = 0; ff_c < 8; ++ff_c) {
    beta_local[ff_c] = ((float*)beta)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.x) * 8)) + ff_c))];
  }
  for (int ff_c_init = 0; ff_c_init < 8; ++ff_c_init) {
    for (int nn_c_init = 0; nn_c_init < 8; ++nn_c_init) {
      B_local[(((ff_c_init * 8) + nn_c_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        __syncthreads();
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
          ((float4*)(Apad_shared + ((((((int)threadIdx.x) * 64) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer * 4)))))[0] = (((((1 <= ((((int)blockIdx.z) / 14) + ry)) && (((((int)blockIdx.z) / 14) + ry) < 15)) && (1 <= (rx + (((int)blockIdx.z) % 14)))) && ((rx + (((int)blockIdx.z) % 14)) < 15)) ? ((float4*)((float*)A + ((((((((((ry * 458752) + (((int)blockIdx.z) * 32768)) + (rx * 32768)) + (rc_outer * 1024)) + (((int)threadIdx.x) * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer * 4)) - 491520))))[0] : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
        }
        for (int ax3_inner_outer1 = 0; ax3_inner_outer1 < 2; ++ax3_inner_outer1) {
          ((float4*)(W_shared + ((((((int)threadIdx.x) * 64) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer1 * 4)))))[0] = ((float4*)((float*)W + ((((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + (((int)threadIdx.x) * 512)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer1 * 4)))))[0];
        }
        __syncthreads();
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          for (int ax3 = 0; ax3 < 8; ++ax3) {
            Apad_shared_local[(ax3)] = Apad_shared[((((rc_inner * 64) + (((int)threadIdx.x) * 8)) + ax3))];
          }
          for (int ax31 = 0; ax31 < 8; ++ax31) {
            W_shared_local[(ax31)] = W_shared[((((rc_inner * 64) + (((int)threadIdx.x) * 8)) + ax31))];
          }
          for (int ff_c = 0; ff_c < 8; ++ff_c) {
            for (int nn_c = 0; nn_c < 8; ++nn_c) {
              B_local[(((ff_c * 8) + nn_c))] = (B_local[(((ff_c * 8) + nn_c))] + ((Apad_shared_local[(nn_c)] * W_shared_local[(ff_c)])));
            }
          }
        }
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 8; ++ff_inner_inner_inner) {
    for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 8; ++nn_inner_inner_inner) {
      ((float*)B)[((((((((((int)blockIdx.z) * 65536) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.x) * 1024)) + (ff_inner_inner_inner * 128)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + nn_inner_inner_inner))] = B_local[(((ff_inner_inner_inner * 8) + nn_inner_inner_inner))] + beta_local[ff_inner_inner_inner];
    }
  }
}

double get_durtime(struct timeval t1, struct timeval t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

int main() {
    const int batch = 128, in_channel = 256, out_channel = 512, in_height = 14, in_width = 14;
    const int kernel_width = 3, kernel_height = 3, pad_height = 1, pad_width = 1;
    const int stride_height = 1, stride_width = 1;
    const int out_width = (in_width - kernel_width + 2*pad_width) / stride_width + 1;
    const int out_height = (in_height - kernel_height + 2*pad_height) / stride_height + 1;
    float *A, *W, *B, *mean, *scale, *var, *beta;
    int in_size = batch*in_channel*in_height*in_width;
    int kernel_size = kernel_width*kernel_height*out_channel*in_channel;
    int out_size = batch*out_channel*out_width*out_height;
    struct timeval start, end;
    A = (float*)malloc(in_size*sizeof(float));
    W = (float*)malloc(kernel_size*sizeof(float));
    B = (float*)malloc(out_size*sizeof(float));
    mean = (float*)malloc(out_channel*sizeof(float));
    scale = (float*)malloc(out_channel*sizeof(float));
    var = (float*)malloc(out_channel*sizeof(float));
    beta = (float*)malloc(out_channel*sizeof(float));
    float *Ad, *Wd, *Bd, *meand, *scaled, *vard, *betad;
    HANDLE_ERROR(cudaMalloc((void**)&Ad, in_size*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Wd, kernel_size*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Bd, out_size*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&meand, out_channel*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&scaled, out_channel*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&vard, out_channel*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&betad, out_channel*sizeof(float)));
    for (int i = 0; i < in_size; ++i)
        A[i] = i;
    for (int i = 0; i < kernel_size; ++i)
        W[i] = i;
    for (int i = 0; i < out_channel; ++i) {
        mean[i] = i;
        scale[i] = i;
        var[i] = i;
        beta[i] = i;
    }
    HANDLE_ERROR(cudaMemcpy(Ad, A, in_size*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(Wd, W, kernel_size*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(meand, mean, out_channel*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(scaled, scale, out_channel*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vard, var, out_channel*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(betad, beta, out_channel*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(2, 8, 196), block(8, 1, 1);
	HANDLE_ERROR(cudaDeviceSynchronize());
    //HANDLE_ERROR(default_function_kernel0<<<grid, block>>>(Ad, Wd, betad, Bd));
    gettimeofday(&start, 0);
    conv_bn_relu<<<grid, block>>>(Ad, Wd, meand, scaled, vard, betad, Bd);
	HANDLE_ERROR(cudaDeviceSynchronize());
    gettimeofday(&end, 0);
    std::cout << "time: " << get_durtime(start, end) << "ms" << std::endl;
    HANDLE_ERROR(cudaMemcpy(B, Bd, out_size*sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(Ad));
    HANDLE_ERROR(cudaFree(Bd));
    HANDLE_ERROR(cudaFree(Wd));
    HANDLE_ERROR(cudaFree(meand));
    HANDLE_ERROR(cudaFree(scaled));
    HANDLE_ERROR(cudaFree(vard));
    HANDLE_ERROR(cudaFree(betad));
    return 0;
}
