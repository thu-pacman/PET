#include "operator.h"
#include "utils.h"
#include <cstdlib>
#include <cassert>
#include <sys/time.h>
#include <iostream>

void Conv2d::performance_measuring(Simulator *simu, int rounds)
{
    if (config.args.size() < 14)
    {
        std::cerr << "invalid configuration of Conv2d!\n";
        exit(0);
    }

    int GROUP_CNT = config.args[0];
    int BATCH_SIZE = config.args[1];
    int IN_CHANNELS_PER_GRP = config.args[2];
    int IN_H = config.args[3];
    int IN_W = config.args[4];
    int OUT_CHANNELS = config.args[5];
    int KN_H = config.args[6];
    int KN_W = config.args[7];
    int PAD_H = config.args[8];
    int PAD_W = config.args[9];
    int STRIDE_H = config.args[10];
    int STRIDE_W = config.args[11];
    int DILA_H = config.args[12];
    int DILA_W = config.args[13];

    int IN_CHANNELS = GROUP_CNT * IN_CHANNELS_PER_GRP;

    double durtime = 0.0, tflops = 0.0;

    for (int i = 0; i < rounds; ++i)
    {
        cudnnHandle_t cudnnHandle;
        CUDNN_CALL(cudnnCreate(&cudnnHandle));

        // get inputs
        cudnnTensorDescriptor_t in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, BATCH_SIZE,
                                              IN_CHANNELS, IN_H, IN_W));

        int in_size = BATCH_SIZE * IN_CHANNELS * IN_H * IN_W;
        float *in_data;
        CUDA_CALL(cudaMalloc(&in_data, in_size * sizeof(float)));

        cudnnFilterDescriptor_t kn_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&kn_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(kn_desc, CUDNN_DATA_FLOAT,
                                              CUDNN_TENSOR_NCHW, OUT_CHANNELS,
                                              IN_CHANNELS_PER_GRP, KN_H, KN_W));

        int kn_size = OUT_CHANNELS * IN_CHANNELS * KN_H * KN_W;
        float *kn_data;
        CUDA_CALL(cudaMalloc(&kn_data, kn_size * sizeof(float)));

        // get convolution descriptor
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            conv_desc, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILA_H, DILA_W,
            CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
        if (GROUP_CNT > 1)
        {
            CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, GROUP_CNT));
        }

        // get ouputs
        int out_n, out_c, out_h, out_w;
        CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, kn_desc, &out_n, &out_c, &out_h, &out_w));
        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                                              out_w));

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
        algo = perfResults[0].algo;

        CUDA_CALL(cudaFree(workSpace));

        // get workspace
        size_t ws_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle, in_desc, kn_desc, conv_desc, out_desc, algo, &ws_size));
        float *ws_data;
        CUDA_CALL(cudaMalloc(&ws_data, ws_size));

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
    tflops = (double)BATCH_SIZE * (double)IN_CHANNELS_PER_GRP *
             (double)OUT_CHANNELS * (double)IN_H * (double)IN_W *
             (double)KN_H * (double)KN_W * 2 / durtime /
             (STRIDE_H * STRIDE_W) / 1000000000.0;

    simu->updatePfMap(config, durtime, tflops, {(int)algo});
}
