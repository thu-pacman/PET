#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cudnn.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
using namespace nvinfer1;
using namespace nvonnxparser;

#define checkCudaError(x)                                                      \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

class Logger : public nvinfer1::ILogger {
  public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
        if (severity == Severity::kVERBOSE)
            return;

        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTENAL ERROR: " << msg << std::endl;
            break;

        case Severity::kERROR:
            std::cerr << "ERROR: " << msg << std::endl;
            break;

        case Severity::kWARNING:
            // std::cerr << "WARNING: " << msg << std::endl;
            break;

        case Severity::kINFO:
            // std::cerr << "INFO: " << msg << std::endl;
            break;

        default:
            break;
        }
    }
};

Logger gLogger;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "./runtrt modelfile.onnx dim0 dim1 dim2\n";
        exit(EXIT_FAILURE);
    }

    const char *filename = argv[1];
    const int inputDims[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};

    IBuilder *builder = createInferBuilder(gLogger);

    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    auto parser = createParser(*network, gLogger);

    auto parsed = parser->parseFromFile(filename, -1);
    if (!parsed) {
        std::cerr << "Error opening flie: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // set the dimension of the input to feed
    const int n = inputDims[0], h = inputDims[1], w = inputDims[2];
    const int inSize = n * h * w;

    auto input = network->getInput(0);
    input->setDimensions(Dims3{n, h, w});
    
    builder->setMaxBatchSize(n);
    // Create an optimization profile and set the dimension as below
    IBuilderConfig *config = builder->createBuilderConfig();
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN,
                           Dims3(n, h, w));
    profile->setDimensions("input", OptProfileSelector::kOPT,
                           Dims3(n, h, w));
    profile->setDimensions("input", OptProfileSelector::kMAX,
                           Dims3(n, h, w));

    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(1UL << 32);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

    // execution
    auto context = engine->createExecutionContext();
    if (!context)
        exit(EXIT_FAILURE);

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));

    void *buffer[2];
    checkCudaError(cudaMalloc(&buffer[0], inSize * sizeof(float)));
    // Allocate enough space for an output
    checkCudaError(cudaMalloc(&buffer[1], inSize * sizeof(float)));
    // checkCudaError(cudaMalloc(&buffer[1], n * 200000 * sizeof(float)));

    float *data = (float*)malloc(inSize * sizeof(float));

    srand(time(nullptr));
    for (int i = 0; i < inSize; ++i) {
        data[i] = rand() % 256 - 128.0;
    }

    checkCudaError(cudaMemcpyAsync(buffer[0], data, inSize * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    context->setBindingDimensions(0, Dims3(n, h, w));

    cudaEvent_t st, ed;
    float duration, sum;
    std::vector<float> samples;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
   
    for (int i = 0; i < 128; i++) {
        checkCudaError(cudaDeviceSynchronize());
        cudaEventRecord(st, 0);    
        bool status = context->executeV2(buffer);
        checkCudaError(cudaDeviceSynchronize());
        cudaEventRecord(ed, 0);
        cudaEventSynchronize(st);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&duration, st, ed);
        samples.emplace_back(duration);	

        for (int i = 0; i < inSize; ++i) {
            data[i] = rand() % 256 - 128.0;
        }   
        checkCudaError(cudaMemcpyAsync(buffer[0], data, inSize * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
        context->setBindingDimensions(0, Dims3(n, h, w));

    }

    std::sort(samples.begin(), samples.end());
    sum = 0;
    for (int i = 32; i < 96; i++) {
        sum += samples[i];
    }
    sum /= 64;
    std::cout << sum << std::endl;

    // Finalize
    engine->destroy();
    checkCudaError(cudaStreamDestroy(stream));
    checkCudaError(cudaFree(buffer[0]));
    checkCudaError(cudaFree(buffer[1]));

    return 0;
}
