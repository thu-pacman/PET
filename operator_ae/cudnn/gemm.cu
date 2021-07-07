#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <sys/time.h>

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf("Cuda error at %s:%d, %d\n", __FILE__, __LINE__,            \
                   EXIT_FAILURE);                                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CALL(x)                                                         \
    do {                                                                       \
        if ((x) != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("Cublas error at %s:%d, %d\n", __FILE__, __LINE__,          \
                   EXIT_FAILURE);                                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CURAND_CALL(x)                                                         \
    do {                                                                       \
        if ((x) != CURAND_STATUS_SUCCESS) {                                    \
            printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

class TestArg {
  public:
    int transa, transb;
    int b, m, n, k;
    float *A, *B, *C;
    int lda, ldb, ldc;

    float test(cublasHandle_t &cublas, int algo) {
        cudaEvent_t st, ed;
        CUDA_CALL(cudaEventCreate(&st));
        CUDA_CALL(cudaEventCreate(&ed));
        float duration = 0.0;
        int warmup = 200, rounds = 1000;
        const float alpha = 1.0, beta = 0.0;
        cublasStatus_t status = cublasGemmStridedBatchedEx(
            cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, m, n,
            k, &alpha, A, CUDA_R_32F, lda, m * k, B, CUDA_R_32F, ldb, k * n,
            &beta, C, CUDA_R_32F, ldc, m * n, b, CUDA_R_32F,
            (cublasGemmAlgo_t)algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Algo: %d failed\n", algo);
            return 10000;
        }
        for (int i = 0; i < warmup; ++i) {
            cublasGemmStridedBatchedEx(
                cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, m,
                n, k, &alpha, A, CUDA_R_32F, lda, m * k, B, CUDA_R_32F, ldb,
                k * n, &beta, C, CUDA_R_32F, ldc, m * n, b, CUDA_R_32F,
                (cublasGemmAlgo_t)algo);
        }
        for (int i = 0; i < rounds; ++i) {
            float durtime;
            CUDA_CALL(cudaEventRecord(st, 0));
            cublasGemmStridedBatchedEx(
                cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, m,
                n, k, &alpha, A, CUDA_R_32F, lda, m * k, B, CUDA_R_32F, ldb,
                k * n, &beta, C, CUDA_R_32F, ldc, m * n, b, CUDA_R_32F,
                (cublasGemmAlgo_t)algo);
            CUDA_CALL(cudaEventRecord(ed, 0));
            CUDA_CALL(cudaEventSynchronize(st));
            CUDA_CALL(cudaEventSynchronize(ed));
            CUDA_CALL(cudaEventElapsedTime(&durtime, st, ed));
	        duration += durtime;
        }
        std::cout << "Algo: " << algo << "\tTimes(ms): " << duration / rounds
                  << std::endl;
        return duration/rounds;
    }
};

void initTestArgs(TestArg *);

int b, m, n, k;
float *d_A, *d_B, *d_C;

int main(int argc, char *argv[]) {
    //scanf("%d%d%d%d", &b, &m, &n, &k);
    if (argc < 5) {
        std::cout << "./bin b m n k" << std::endl;
        return 1;
    }
    b = atoi(argv[1]);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    k = atoi(argv[4]);

    CUDA_CALL(cudaMalloc((void **)&d_A, b * m * k * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_B, b * k * n * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_C, b * m * n * sizeof(float)));

    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock()));

    CURAND_CALL(curandGenerateUniform(gen, d_A, b * m * k));
    CURAND_CALL(curandGenerateUniform(gen, d_B, b * k * n));

    cublasHandle_t cublas;
    CUBLAS_CALL(cublasCreate(&cublas));

    TestArg Test[8];
    initTestArgs(Test);

    int bestalgo = -2, bestmode = -1;
    double besttime = 10000;
    for (int i = 0; i < 8; ++i) {
        for (int j = -1; j < 24; ++j) {
            printf("Num: %d\t", i);
            double t = Test[i].test(cublas, j);
            if (t < besttime) {
                bestalgo = j;
                bestmode = i;
                besttime = t;
            }
        }
    }
    std::cout << "best algo: " << bestalgo << ", best mode: " << bestmode << ", best time: " << besttime << std::endl;

    // Finalize
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
    CUBLAS_CALL(cublasDestroy(cublas));
    CURAND_CALL(curandDestroyGenerator(gen));

    return 0;
}

void initTestArgs(TestArg *Test) {
    // d_C = d_A x d_B
    Test[0].transa = 0;
    Test[0].transb = 0;
    Test[0].b = b;
    Test[0].A = d_A;
    Test[0].lda = m;
    Test[0].B = d_B;
    Test[0].ldb = k;
    Test[0].C = d_C;
    Test[0].ldc = m;
    Test[0].m = m;
    Test[0].n = n;
    Test[0].k = k;

    Test[1].transa = 1;
    Test[1].transb = 0;
    Test[1].b = b;
    Test[1].A = d_A;
    Test[1].lda = k;
    Test[1].B = d_B;
    Test[1].ldb = k;
    Test[1].C = d_C;
    Test[1].ldc = m;
    Test[1].m = m;
    Test[1].n = n;
    Test[1].k = k;

    Test[2].transa = 0;
    Test[2].transb = 1;
    Test[2].b = b;
    Test[2].A = d_A;
    Test[2].lda = m;
    Test[2].B = d_B;
    Test[2].ldb = n;
    Test[2].C = d_C;
    Test[2].ldc = m;
    Test[2].m = m;
    Test[2].n = n;
    Test[2].k = k;

    Test[3].transa = 1;
    Test[3].transb = 1;
    Test[3].b = b;
    Test[3].A = d_A;
    Test[3].lda = k;
    Test[3].B = d_B;
    Test[3].ldb = n;
    Test[3].C = d_C;
    Test[3].ldc = m;
    Test[3].m = m;
    Test[3].n = n;
    Test[3].k = k;

    // trans(d_C) = trans(d_B) x trans(d_A)
    Test[4].transa = 1;
    Test[4].transb = 1;
    Test[4].b = b;
    Test[4].A = d_B;
    Test[4].lda = k;
    Test[4].B = d_A;
    Test[4].ldb = m;
    Test[4].C = d_C;
    Test[4].ldc = n;
    Test[4].m = n;
    Test[4].n = m;
    Test[4].k = k;

    Test[5].transa = 0;
    Test[5].transb = 1;
    Test[5].b = b;
    Test[5].A = d_B;
    Test[5].lda = n;
    Test[5].B = d_A;
    Test[5].ldb = m;
    Test[5].C = d_C;
    Test[5].ldc = n;
    Test[5].m = n;
    Test[5].n = m;
    Test[5].k = k;

    Test[6].transa = 1;
    Test[6].transb = 0;
    Test[6].b = b;
    Test[6].A = d_B;
    Test[6].lda = k;
    Test[6].B = d_A;
    Test[6].ldb = k;
    Test[6].C = d_C;
    Test[6].ldc = n;
    Test[6].m = n;
    Test[6].n = m;
    Test[6].k = k;

    Test[7].transa = 0;
    Test[7].transb = 0;
    Test[7].b = b;
    Test[7].A = d_B;
    Test[7].lda = n;
    Test[7].B = d_A;
    Test[7].ldb = k;
    Test[7].C = d_C;
    Test[7].ldc = n;
    Test[7].m = n;
    Test[7].n = m;
    Test[7].k = k;
}
