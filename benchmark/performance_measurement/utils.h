#ifndef UTILS_H
#define UTILS_H

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

#define CUDNN_CALL(x) do { if((x) != CUDNN_STATUS_SUCCESS) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
        printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE); \
        exit(EXIT_FAILURE);}} while(0)

#include <sys/time.h>
#include <ctime>
#include <curand.h>
#include <cstdio>

double get_durtime(struct timeval beg, struct timeval end) {
    double t = (1000000.0 * (end.tv_sec - beg.tv_sec) + end.tv_usec - beg.tv_usec) / 1000.0;
    //printf("Elapse: %f ms\n", t);
	return t;
}

void rand_gen_data(float* des, int num_of_elems) {
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long) clock()));
    CURAND_CALL(curandGenerateUniform(generator, des, num_of_elems));
}

#endif
