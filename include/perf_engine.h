#pragma once

#include "common.h"
#include "operator.h"
#include "perf.h"
#include <cstdint>
#include <cuda.h>
#include <map>

namespace tpm {

class PerfEngine {
  private:
    int penaltyFlag = 1;
    std::map<ConvArgs, ConvResult> convPerf;
    std::map<MatmulArgs, MatmulResult> matmulPerf;
    std::map<PoolArgs, float> maxPoolPerf;
    std::map<PoolArgs, float> avgPoolPerf;

    // for ConvOp
    float *inputPtr;
    float *weightPtr;
    float *outputPtr;
    float *biasPtr;
    float *workspace;

    // for MatmulOp
    float *matA;
    float *matB;
    float *matC;

    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    void allocMem();

  public:
    PerfEngine() {
        allocMem();
        checkCudnnError(cudnnCreate(&cudnn));
        checkCublasError(cublasCreate(&cublas));
    }

    ~PerfEngine() {
        dumpPerfData();
        checkCudaError(cudaFree(inputPtr));
        checkCudaError(cudaFree(weightPtr));
        checkCudaError(cudaFree(outputPtr));
        checkCudaError(cudaFree(workspace));
        checkCudnnError(cudnnDestroy(cudnn));
        checkCublasError(cublasDestroy(cublas));
    }

    void setPenalty(int flag) { penaltyFlag = flag; }
    int withPenalty() const { return penaltyFlag; }

    float *getInputPtr() { return inputPtr; }
    float *getWeightPtr() { return weightPtr; }
    float *getBiasPtr() { return biasPtr; }
    float *getOutputPtr() { return outputPtr; }
    float *getWorkspace() { return workspace; }
    float *getMatA() { return matA; }
    float *getMatB() { return matB; }
    float *getMatC() { return matC; }
    cudnnHandle_t cudnnHandle() const { return cudnn; }
    cublasHandle_t cublasHandle() const { return cublas; }

    cudnnConvolutionFwdAlgo_t getConvAlgo(const ConvArgs &args) {
        return convPerf.at(args).algo;
    }
    cublasGemmAlgo_t getMatmulAlgo(const MatmulArgs &args) {
        return matmulPerf.at(args).algo;
    }

    template <class OpArgs>
    double getOpPerf(Operator::OpType opType, const OpArgs &args) {
        return 0.0;
    }
    double getOpPerf(Operator::OpType opType, const ConvArgs &args) {
        return convPerf.at(args).time;
    }
    double getOpPerf(Operator::OpType opType, const MatmulArgs &args) {
        return matmulPerf.at(args).time;
    }

    template <class OpArgs>
    bool checkOpPerf(Operator::OpType opType, const OpArgs &args) {
        return true;
    }
    bool checkOpPerf(Operator::OpType opType, const ConvArgs &args) {
        return convPerf.count(args);
    }
    bool checkOpPerf(Operator::OpType opType, const MatmulArgs &args) {
        return matmulPerf.count(args);
    }

    void saveOpPerf(uint32_t opType, const ConvArgs &args,
                    const ConvResult &perf) {
        convPerf[args] = perf;
    }
    void saveOpPerf(uint32_t opType, const MatmulArgs &args,
                    const MatmulResult &perf) {
        matmulPerf[args] = perf;
    }

    void saveOpPerf(uint32_t opType, const PoolArgs &args, const float perf) {
        if (opType == Operator::MaxPool)
            maxPoolPerf[args] = perf;
        else if (opType == Operator::AvgPool)
            avgPoolPerf[args] = perf;
        else
            assert(0);
    }

    void dumpPerfData();
};

} // namespace tpm
