#include "operator.h"
#include "simulator.h"
#include "utils.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cudnn.h>
#include <iostream>
#include <sys/time.h>

Conv2d::Conv2d(const std::string &id, int group, int batch_size, int in_channel,
               int out_channel, int in_h, int in_w, int kn_h, int kn_w,
               int pad_h, int pad_w, int stride_h, int stride_w, int dila_h,
               int dila_w) {
  GROUP = group;
  BATCH_SIZE = batch_size;
  IN_CHANNEL = in_channel;
  OUT_CHANNEL = out_channel;
  IN_H = in_h;
  IN_W = in_w;
  KN_H = kn_h;
  KN_W = kn_w;
  PAD_H = pad_h;
  PAD_W = pad_w;
  STRIDE_H = stride_h;
  STRIDE_W = stride_w;
  DILA_H = dila_h;
  DILA_W = dila_w;
}

/*========================================================================================================*/
// Performance measuring

static const char algo_name[8][50] = {
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
};

void Conv2d::performance_measuring(Simulator *simu, int rounds) {
  double durtime = 0.0;
  double tflops = 0.0;

  std::cout << "input shape: [" << BATCH_SIZE << ", " << IN_CHANNEL << ", "
            << IN_H << ", " << IN_W << "]\n"
            << "kernel shape: [" << OUT_CHANNEL << ", " << IN_CHANNEL << ", "
            << KN_H << ", " << KN_W << "]\n";

  for (int i = 0; i < rounds; ++i) {

    cudnnHandle_t cudnnHandle;
    CUDNN_CALL(cudnnCreate(&cudnnHandle));

    // get inputs
    cudnnTensorDescriptor_t in_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, BATCH_SIZE,
                                          IN_CHANNEL, IN_H, IN_W));

    int in_size = BATCH_SIZE * IN_CHANNEL * IN_H * IN_W;
    float *in_data;
    CUDA_CALL(cudaMalloc(&in_data, in_size * sizeof(float)));

    // get kernels
    cudnnFilterDescriptor_t kn_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&kn_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(kn_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, OUT_CHANNEL,
                                          IN_CHANNEL, KN_H, KN_W));

    int kn_size = OUT_CHANNEL * IN_CHANNEL * KN_H * KN_W;
    float *kn_data;
    CUDA_CALL(cudaMalloc(&kn_data, kn_size * sizeof(float)));

    // get convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILA_H, DILA_W,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    if (GROUP > 1) {
      CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, GROUP));
    }

    // get ouputs
    int out_n, out_c, out_h, out_w;
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, kn_desc, &out_n, &out_c, &out_h, &out_h));
    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                                          out_h));

    int out_size = out_n * out_c * out_h * out_w;
    float *out_data;
    CUDA_CALL(cudaMalloc(&out_data, out_size * sizeof(float)));

    // get default workspace
    size_t workSpaceSize = (size_t)1024 * 1024 * 1024;
    float *workSpace;
    CUDA_CALL(cudaMalloc(&workSpace, workSpaceSize));

    // get algorithm
    const int conv_algo_cnt = 8;
    int cnt = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[conv_algo_cnt];
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(
        cudnnHandle, in_desc, in_data, kn_desc, kn_data, conv_desc, out_desc,
        out_data, conv_algo_cnt, &cnt, perfResults, workSpace, workSpaceSize));
    assert(cnt > 0);
    CUDNN_CALL(perfResults[0].status);
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;

    CUDA_CALL(cudaFree(workSpace));

    // get workspace
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle, in_desc, kn_desc, conv_desc, out_desc, algo, &ws_size));
    float *ws_data;
    CUDA_CALL(cudaMalloc(&ws_data, ws_size));

    if (i == 0) {
      std::cout << "output shape: [" << out_n << ", " << out_c << ", " << out_h
                << ", " << out_w << "]\n"
                << "Algoritm: " << algo_name[algo] << std::endl
                << "Workspace size: " << ws_size << " bytes\n";
    }

    // perform convolution
    float alpha = 1.f;
    float beta = 0.f;

    struct timeval beg, end;
    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&beg, 0);
    CUDNN_CALL(cudnnConvolutionForward(
        cudnnHandle, &alpha, in_desc, in_data, kn_desc, kn_data, conv_desc,
        algo, ws_data, ws_size, &beta, out_desc, out_data));
    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&end, 0);
    durtime += get_durtime(beg, end);

    // finalize
    CUDA_CALL(cudaFree(in_data));
    CUDA_CALL(cudaFree(kn_data));
    CUDA_CALL(cudaFree(out_data));
    CUDA_CALL(cudaFree(ws_data));

    CUDNN_CALL(cudnnDestroy(cudnnHandle));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(kn_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  }

  durtime /= rounds;
  tflops = (double)BATCH_SIZE * (double)IN_CHANNEL * (double)OUT_CHANNEL *
           (double)IN_H * (double)IN_W * (double)KN_H * (double)KN_W * 2 /
           durtime / (STRIDE_H * STRIDE_W) / 1000000000.0;

  std::cout << "durtime: " << durtime << " ms\n"
            << "tflops: " << tflops << std::endl;

  perfInfo.update(durtime, tflops);
}
