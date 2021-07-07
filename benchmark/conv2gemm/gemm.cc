// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/time.h>
#include <stdio.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
        printf("Error at %s:%d, %d\n",__FILE__,__LINE__, EXIT_FAILURE); \
        return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
        printf("Error at %s:%d, %d\n",__FILE__,__LINE__, EXIT_FAILURE); \
        return EXIT_FAILURE;}} while(0)

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

double get_durtime(struct timeval t1, struct timeval t2) {
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	struct timeval start, end;
	
	gettimeofday(&start, 0);
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	gettimeofday(&end, 0);
  double t = get_durtime(start, end);
	std::cout << "time: " << t << "ms" << std::endl;
  std::cout << "FLOPS: " << 2.0*m*n*k*1000/t/1e12 << " tflops" << std::endl;

	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    int input_n, input_c, input_h, input_w;
    int kernel_h, kernel_w;
    int output_c, output_h, output_w;

    if (argc < 8) {
        std::cout << "args: input_n input_c input_h input_w kernel_h kernel_w output_c" << std::endl;
        return 1;
    }

    input_n = atoi(argv[1]);
    input_c = atoi(argv[2]);
    input_h = atoi(argv[3]);
    input_w = atoi(argv[4]);
    kernel_h = atoi(argv[5]);
    kernel_w = atoi(argv[6]);
    output_c = atoi(argv[7]);
    output_h = input_h - kernel_h + 1;
    output_w = input_w - kernel_w + 1;

    // Allocate arrays on GPU
    float* d_input, *d_kernel, *d_output;
    CUDA_CALL(cudaMalloc(&d_input, input_n * input_c * input_h * input_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_kernel, input_c * output_c * kernel_h * kernel_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_output, input_n * output_c * output_h * output_w * sizeof(float)));

    // Fill the arrays with random numbers
    GPU_fill_rand(d_input, input_n * input_c, input_h * input_w);
    GPU_fill_rand(d_kernel, input_c * output_c, kernel_h * kernel_w);

    struct timeval start, end;
    // gettimeofday(&start, 0);

//// Shuffle add
    float* d_kernel_tmp, *d_output_tmp;
    CUDA_CALL(cudaMalloc(&d_kernel_tmp, output_c * input_c * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_output_tmp, input_n * output_c * input_h * input_w * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    float* d_input_tmp, *base;
    cudaMalloc(&d_input_tmp, input_n * (input_c+1) * input_h * input_w * sizeof(float));
    CUBLAS_CALL(cublasScopy(handle, input_n * input_c * input_h * input_w, d_input, 1, d_input_tmp, 1));

    const int m = output_c, n = input_h * input_w * input_n, k = input_c;
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    float bet = 0;
    const float* alpha = &alf;
    float* beta = &bet;

    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&start, 0);
    // For the (0, 0) element in the kernel
    CUBLAS_CALL(cublasScopy(handle, output_c * input_c, d_kernel, kernel_h * kernel_w, d_kernel_tmp, 1));
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_kernel_tmp, lda, d_input, ldb, beta, d_output_tmp, ldc));

    // For other elements in the kernel
    bet = 1;
    for (int i = 0; i < kernel_h; ++ i) {
        int j = (i == 0) ? 1 : 0;
        for (; j < kernel_w; ++ j) {
            base = d_input_tmp + i * input_w + j;
            CUBLAS_CALL(cublasScopy(handle, output_c * input_c, d_kernel + i * kernel_w + j, kernel_h * kernel_w, d_input_tmp, 1));
            CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_kernel_tmp, lda, base, ldb, beta, d_output_tmp, ldc));
        }
    }

    for (int i = 0; i < output_w; ++ i) {
        CUBLAS_CALL(cublasScopy(handle, output_h, d_output_tmp + i, input_w, d_output + i, output_w));
    }

    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&end, 0);
    std::cout << "time: " << get_durtime(start, end) << "ms" << std::endl;

    CUBLAS_CALL(cublasDestroy(handle));

    // Free GPU memory
    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_kernel));
    CUDA_CALL(cudaFree(d_output));
    CUDA_CALL(cudaFree(d_input_tmp));
    CUDA_CALL(cudaFree(d_kernel_tmp));
    CUDA_CALL(cudaFree(d_output_tmp));

    return 0;
}
