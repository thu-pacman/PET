#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cudnn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include "utils.h"

/* <====== configuration for the test of convolution ======> */

#define PAD 1

class Config {

private:
    int batch_size, input_channel, output_channel, input_h, input_w, kernel_h, kernel_w;
    
    // rows(columns) of zeros implicitly concatenated onto the input maps
    int pad_h, pad_w;
    
    int stride_h, stride_w;
    int dilation_h, dilation_w;

public:
    Config() {
    // modify params here
    
        batch_size = 64;
        input_channel = 960;
        output_channel = 960;
        input_h = 7;
        input_w = 7;
        kernel_h = 3;
        kernel_w = 3;
        
        pad_h = 0;
        pad_w = 0;
        
        stride_h = 1;
        stride_w = 1;
        
        dilation_h = 1;
        dilation_w = 1;
    }

    Config(const Config& other) {
        this->batch_size = other.batch_size;
        this->input_channel = other.input_channel;
        this->output_channel = other.output_channel;
        this->input_h = other.input_h;
        this->input_w = other.input_w;

        this->pad_h = other.pad_h;
        this->pad_w = other.pad_w;
        this->stride_h = other.stride_h;
        this->stride_w = other.stride_w;

        this->dilation_h = other.dilation_h;
        this->dilation_w = other.dilation_w;
    }

    Config& operator=(const Config& other) {
        this->batch_size = other.batch_size;
        this->input_channel = other.input_channel;
        this->output_channel = other.output_channel;
        this->input_h = other.input_h;
        this->input_w = other.input_w;

        this->pad_h = other.pad_h;
        this->pad_w = other.pad_w;
        this->stride_h = other.stride_h;
        this->stride_w = other.stride_w;

        this->dilation_h = other.dilation_h;
        this->dilation_w = other.dilation_w;

    }
    
    friend class Convolution;
    friend void search(Config originConfig, int level, int layoutchange_sum, std::vector<int>* sum_set);
    
    // methods for datalayout change
    
    /* (n,c,h,w) -> (n/2,c,2*h+pad,w) */
    Config nDiv2h() {
        Config ret = *this;
        ret.batch_size = batch_size / 2;
        ret.input_h = 2 * input_h + PAD;
        
        return ret;
    }
    
    /* (n,c,h,w) -> (n/2,c,h,2*w+pad) */
    Config nDiv2w() {
        Config ret = *this;
        ret.batch_size = batch_size / 2;
        ret.input_w = 2 * input_w + PAD;
        
        return ret;
    }
    
    /* (n,c,h,w) -> (n/4,c,2*h+pad,2*w+pad) */
    Config nDiv4() {
        Config ret = *this;
        ret.batch_size = batch_size / 4;
        ret.input_h = 2 * input_h + PAD;
        ret.input_w = 2 * input_w + PAD;
        
        return ret;
    }
    
    /* (n,c,h,w) -> (2*n,c,h/2+pad,w) */
    Config nMul2h() {
        Config ret = *this;
        ret.batch_size = batch_size * 2;
        ret.input_h = input_h / 2 + PAD;
        
        return ret;
    }
    
    /* (n,c,h,w) -> (2*n,c,h,w/2+pad) */
    Config nMul2w() {
        Config ret = *this;
        ret.batch_size = batch_size * 2;
        ret.input_w = input_w / 2 + PAD;
        
        return ret;
    }
    
    /* (n,c,h,w) -> (4*n,c,h/2+pad,w/2+pad) */
    Config nMul4() {
        Config ret = *this;
        ret.batch_size = batch_size * 4;
        ret.input_h = input_h / 2 + PAD;
        ret.input_w = input_w / 2 + PAD;
        
        return ret;
    }
};
/*----------------------------------------------------------------------------*/

/* <====== convlution based on cudnn ======> */

class Convolution {

private:
    Config config;
    
public:
    Convolution() {}
    
    Convolution(const Config& conf) {
        config = conf;
    }
    
    void test() {
        cudnnHandle_t cudnnHandle;
        CUDNN_CALL(cudnnCreate(&cudnnHandle));
        
        // get inputs
        cudnnTensorDescriptor_t input_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CALL(
            cudnnSetTensor4dDescriptor(
                input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                config.batch_size,
                config.input_channel,
                config.input_h,
                config.input_w )
        );
        
        float* input_data;
        CUDA_CALL(cudaMalloc(&input_data, config.batch_size * config.input_channel * config.input_h * config.input_w * sizeof(float)));
        rand_gen_data(input_data, config.batch_size * config.input_channel * config.input_h * config.input_w);

        // get kernels
        cudnnFilterDescriptor_t kernel_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&kernel_desc));
        CUDNN_CALL(
            cudnnSetFilter4dDescriptor(
                kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                config.output_channel,
                config.input_channel,
                config.kernel_h,
                config.kernel_w )
        );
        
        float* kernel_data;
        CUDA_CALL(cudaMalloc(&kernel_data, config.output_channel * config.input_channel * config.kernel_h * config.kernel_w * sizeof(float)));
        rand_gen_data(kernel_data, config.output_channel * config.input_channel * config.kernel_h * config.kernel_w);

        // get output
        int output_n, output_c, output_h, output_w;
        
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
                    conv_desc, config.pad_h, config.pad_w, config.stride_h, config.stride_w,
                    config.dilation_h, config.dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        CUDNN_CALL(
            cudnnGetConvolution2dForwardOutputDim(
                conv_desc, input_desc, kernel_desc,
                &output_n, &output_c, &output_h, &output_w )
        );

        cudnnTensorDescriptor_t output_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CALL(
            cudnnSetTensor4dDescriptor(
                output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                output_n, output_c, output_h, output_w )
        );
        
        float* output_data;
        CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));
        
        // get algorithm
        cudnnConvolutionFwdAlgo_t algorithm;
        CUDNN_CALL(
            cudnnGetConvolutionForwardAlgorithm(
                cudnnHandle,
                input_desc, kernel_desc, conv_desc, output_desc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                0, &algorithm)
        );
        
        std::cout << "algorithm: " << algorithm << std::endl;
        
        // get workspace
        size_t ws_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle, input_desc, kernel_desc, conv_desc, output_desc, algorithm, &ws_size));

        float *ws_data;
        CUDA_CALL(cudaMalloc(&ws_data, ws_size));

        // perform convolution
        float alpha = 1.f;
        float beta = 0.f;
 
        struct timeval beg, end;
        CUDA_CALL(cudaDeviceSynchronize());
        gettimeofday(&beg, 0);
        CUDNN_CALL(cudnnConvolutionForward(
            cudnnHandle,
            &alpha, input_desc, input_data, kernel_desc, kernel_data,
            conv_desc, algorithm, ws_data, ws_size,
            &beta, output_desc, output_data));
        CUDA_CALL(cudaDeviceSynchronize());
        gettimeofday(&end, 0);
        print_durtime(beg, end);
        
        // finalize
        CUDA_CALL(cudaFree(input_data));
        CUDA_CALL(cudaFree(kernel_data));
        CUDA_CALL(cudaFree(output_data));
        CUDA_CALL(cudaFree(ws_data));
        
        CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CALL(cudnnDestroyFilterDescriptor(kernel_desc));
        CUDNN_CALL(cudnnDestroy(cudnnHandle));
        CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    }

};

/*----------------------------------------------------------------------------*/

/* <====== search ======> */

/*
 in order to avoid visiting the same
 situation, allocate a designed set of
 integers for those methods of datalayout change,
 please go through the diagram in https://
 */

void search(Config originConfig, int level, int layoutchange_sum, std::vector<int>* sum_set) {
    if (level >= 5) return;
    if (originConfig.batch_size <= 15) return;
    if (originConfig.input_h < 4 || originConfig.input_w < 4) return;
    
    sum_set->push_back(layoutchange_sum);
    printf("[%d, %d, %d, %d]: ", originConfig.batch_size, originConfig.input_channel, originConfig.input_h, originConfig.input_w);
    Convolution* conv = new Convolution(originConfig);
    conv->test();
    delete conv;
    
    static int layout_change[6] = {2, -2, 17, -17, 19, -19};
    for (int i = 0; i < 6; ++ i) {
        if (find(sum_set->begin(), sum_set->end(), layoutchange_sum + layout_change[i]) != sum_set->end()) {
            continue;
        }
        int id = layout_change[i];
        switch (id) {
            case -19:
                search(originConfig.nMul4(), layoutchange_sum-19, level+1, sum_set);
                break;
                
            case -17:
                search(originConfig.nMul2h(), layoutchange_sum-17, level+1, sum_set);
                break;
                
            case -2:
                search(originConfig.nMul2w(), layoutchange_sum-2, level+1, sum_set);
                break;
                
            case 2:
                search(originConfig.nDiv2w(), layoutchange_sum+2, level+1, sum_set);
                break;
                
            case 17:
                search(originConfig.nDiv2h(), layoutchange_sum+17, level+1, sum_set);
                break;
                
            case 19:
                search(originConfig.nDiv4(), layoutchange_sum+19, level+1, sum_set);
                break;
                
            default:
                break;
        }
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> sum_set;
    Config config = Config();
    search(config, 0, 0, &sum_set);
    
    return 0;
}
