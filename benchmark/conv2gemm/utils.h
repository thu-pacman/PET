#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <ctime>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

void print_tensor(const float* h_ts, int d1, int d2, int d3, int d4) {
    for (int i = 0; i < d1; ++ i) {
        for (int j = 0; j < d2; ++ j) {
            for (int k = 0; k < d3; ++ k) {
                for (int m = 0; m < d4; ++ m) {
                    printf("%.4f ", h_ts[i * d2*d3*d4 + j * d3*d4 + k * d4 + m]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void read_tensors_from_file(const char* fileName,
                            int* bat_sz,
                            int* inp_c, int* oup_c,
                            int* inp_h, int* inp_w,
                            int* knl_h, int* knl_w,
                            int* oup_h, int* oup_w,
                            float** d_knl, float** d_inp)
{
    int N, iC, iH, iW, kH, kW, oC;
    FILE* fp = fopen(fileName, "r");
    fscanf(fp, "%d%d%d%d%d%d%d",
           &N, &iC,
           &iH, &iW,
           &kH, &kW,
           &oC);
    
    int knl_len, inp_len;
    knl_len = oC * iC * kH * kW;
    inp_len = iC * N * iH * iW;
    
    // Host memory
    float* h_knl, *h_inp;
    h_knl = (float*)malloc(knl_len * sizeof(float));
    h_inp = (float*)malloc(inp_len * sizeof(float));

    // Device memory
    CUDA_CALL(cudaMalloc(d_knl, knl_len * sizeof(float)));
    CUDA_CALL(cudaMalloc(d_inp, (inp_len + iH * iW) * sizeof(float)));

    // Read filters
    for (int i = 0; i < knl_len; ++ i)
        fscanf(fp, "%f", &h_knl[i]);
    // Read inputs
    for (int i = 0; i < inp_len; ++ i)
        fscanf(fp, "%f", &h_inp[i]);

    // print_tensor(h_knl, kH, kW, oC, iC);
    // print_tensor(h_inp, iC, N, iH, iW);

    fclose(fp);

    CUDA_CALL(cudaMemcpy(*d_knl, h_knl, knl_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(*d_inp, h_inp, inp_len * sizeof(float), cudaMemcpyHostToDevice));

    free(h_knl);
    free(h_inp);

    *bat_sz = N;
    *inp_c = iC; *oup_c = oC;
    *inp_h = iH; *inp_w = iW;
    *knl_h = kH; *knl_w = kW;
    *oup_h = iH - kH + 1;
    *oup_w = iW - kW + 1;
}

void randn_gen_tensors(int argc, char* argv[],
                       int* bat_sz,
                       int* inp_c, int* oup_c,
                       int* inp_h, int* inp_w,
                       int* knl_h, int* knl_w,
                       int* oup_h, int* oup_w,
                       float** d_inp, float** d_knl)
{
    if (argc < 8) {
        printf("args: batch_size input_c output_c input_h input_w kernel_h kernel_w\n");
            exit(1);
    }
            
    int bs, ic, oc, ih, iw, kh, kw, oh, ow;
    bs = atoi(argv[1]);
    ic = atoi(argv[2]);
    oc = atoi(argv[3]);
    ih = atoi(argv[4]);
    iw = atoi(argv[5]);
    kh = atoi(argv[6]);
    kw = atoi(argv[7]);
    oh = ih - kh + 1;
    ow = iw - kw + 1;
                                                    
    int inp_len, knl_len;
    inp_len = bs * ic * ih * iw;
    knl_len = oc * ic * kh * kw;
                                                                
    CUDA_CALL(cudaMalloc(d_inp, (inp_len + ih * iw) * sizeof(float))); // Add a buffer of length ih*iw at the end of the array of inputs
    CUDA_CALL(cudaMalloc(d_knl, knl_len * sizeof(float)));

    *bat_sz = bs;
    *inp_c = ic; *oup_c = oc;
    *inp_h = ih; *inp_w = iw;
    *knl_h = kh; *knl_w = kw;
    *oup_h = oh; *oup_w = ow;

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long) clock()));
    CURAND_CALL(curandGenerateUniform(generator, *d_inp, inp_len));
    CURAND_CALL(curandGenerateUniform(generator, *d_knl, knl_len));
                                                                                                                
}


void free_tensors(float* d_inp, float* d_knl) {
    CUDA_CALL(cudaFree(d_inp));
    CUDA_CALL(cudaFree(d_knl));
}

#endif
