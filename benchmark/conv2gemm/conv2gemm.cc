#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include "utils.h"
#include <iostream>


double durtime(struct timeval beg, struct timeval end) {
    double t = (1000000.0 * (end.tv_sec - beg.tv_sec) + end.tv_usec - beg.tv_usec) / 1000.0;
    return t;
}

void shuffle_add(float* d_knl, float* d_inp, float* d_oup,
                 int bat_sz,
                 int inp_c, int oup_c,
                 int inp_h, int inp_w,
                 int knl_h, int knl_w,
                 int oup_h, int oup_w)
{
    const int M = knl_h * knl_w * oup_c, N = bat_sz * inp_h * inp_w, K = inp_c;
    const int lda = M, ldb = K, ldc = M;
    const float alf = 1, bet = 0;
    const float* alpha = &alf, *beta = &bet;
    
    float* d_tmp;
    CUDA_CALL(cudaMalloc(&d_tmp, M * N * sizeof(float)));
    
    int stride = oup_c * N;
    int vec_len = stride - ((knl_h -1) * inp_w + knl_w - 1);
    float* y = d_tmp, *x;
    int incx = 1, incy = 1;
    
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    
    struct timeval beg, end;
    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&beg, 0);
    
    CUBLAS_CALL(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            M, N, K,
                            alpha,
                            d_knl, lda,
                            d_inp, ldb,
                            beta,
                            d_tmp, ldc)
                );
    
    for (int i = 0; i < knl_h; ++ i) {
        int j = (i == 0) ? 1 : 0;
        for ( ; j < knl_w; ++ j) {
            x = d_tmp + (i*knl_w+j) * stride + i*inp_w + j;
            CUBLAS_CALL(cublasSaxpy(handle, vec_len,
                                    alpha,
                                    x, incx,
                                    y, incy)
                        );
        }
    }
    
    CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&end, 0);
    
    double t = durtime(beg, end);
    std::cout << "time: " << t << " ms" << std::endl;
    std::cout << "gemm shape: m, n, k = " << M << ", " << N << ", " << K << std::endl;
    std::cout << "flops: " << 2.0*M*N*K/t/1e9 << " tflops" << std::endl;
    
    CUBLAS_CALL(cublasDestroy(handle));
    CUDA_CALL(cudaFree(d_tmp));
}

int main(int argc, char* argv[]) {
    float* d_knl, *d_inp;
    int bat_sz, inp_c, oup_c, inp_h, inp_w, knl_h, knl_w, oup_h, oup_w;
#ifdef DEBUG 
    if (argc < 2) {
        printf("args: file name\n");
        exit(1);
    }

    read_tensors_from_file(argv[1],
                           &bat_sz,
                           &inp_c, &oup_c,
                           &inp_h, &inp_w,
                           &knl_h, &knl_w,
                           &oup_h, &oup_w,
                           &d_knl, &d_inp
                           );
#else
    randn_gen_tensors(argc, argv,
                      &bat_sz,
                      &inp_c, &oup_c,
                      &inp_h, &inp_w,
                      &knl_h, &knl_w,
                      &oup_h, &oup_w,
                      &d_inp, &d_knl
                      );
#endif
    float* d_oup;
    CUDA_CALL(cudaMalloc(&d_oup, oup_c * bat_sz * oup_h * oup_w * sizeof(float)));

    shuffle_add(d_knl, d_inp, d_oup,
                bat_sz,
                inp_c, oup_c,
                inp_h, inp_w,
                knl_h, knl_w,
                oup_h, oup_w
                );

    CUDA_CALL(cudaFree(d_oup));
    free_tensors(d_inp, d_knl);

    return 0;
}
