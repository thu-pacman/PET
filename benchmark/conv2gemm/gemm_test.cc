#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <curand.h>
#include <cublas_v2.h>
#include "utils.h"

void print_durtime(struct timeval beg, struct timeval end) {
    double t = (1000000.0 * (end.tv_sec - beg.tv_sec) + end.tv_usec - beg.tv_usec) / 1000.0;
    printf("Elapse: %f ms\n", t);
}

void randn_gen_mat(float* des, int len) {
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long) clock()));
    CURAND_CALL(curandGenerateUniform(generator, des, len));
}

void cublas_gemm(const float* A, const float* B, float* C, int batch, int m, int n, int k) {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    const int lda = m, ldb = k, ldc = m;
    const float alf = 1.0, bet = 0.0;
    const float* alpha = &alf;
    const float* beta = &bet;

    struct timeval beg, end;
    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&beg, 0);

    for (int i = 0; i < batch; ++ i) {
        CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A + i * m * k, lda, B + i * k * n, ldb, beta, C + i * m * n, ldc));
    }

    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&end, 0);
    
    printf("Gemm ");
    print_durtime(beg, end);
}

void cublas_batch_gemm(const float* A, const float* B, float* C, int batch, int m, int n, int k) {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    const int lda = m, ldb = k, ldc = m;
    const float alf = 1.0, bet = 0.0;
    const float* alpha = &alf;
    const float* beta = &bet;

    struct timeval beg, end;
    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&beg, 0);

    CUBLAS_CALL(cublasSgemmStridedBatched(handle, CUBALS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, m * k, B, ldb, k * n, beta, C, ldc, m * n, batch));

    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&end, 0);

    printf("Batch Gemm ");
    print_durtime(beg, end);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("args: batch m k n\n");
        return 1;
    }

    int batch, m, k, n;
    batch = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);
    n = atoi(argv[4]);

    float* A;
    float* B;
    float* C;
    CUDA_CALL(cudaMalloc(&A, batch * m * k * sizeof(float)));
    CUDA_CALL(cudaMalloc(&B, batch * k * n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&C, batch * m * n * sizeof(float)));

    randn_gen_mat(A, batch * m * k);
    randn_gen_mat(B, batch * k * n);

    // test for gemm
    cublas_gemm(A, B, C, batch, m, n, k);

    // test for batch gemm

    // free device memory
    CUDA_CALL(cudaFree(A));
    CUDA_CALL(cudaFree(B));
    CUDA_CALL(cudaFree(C));

    return 0;
}