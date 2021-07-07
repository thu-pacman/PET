#include <cassert>
#include <chrono>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <sys/time.h>

typedef float d_type;
#define DATA_TYPE CUDNN_DATA_FLOAT

int GROUP_COUNT = 1;
int INPUT_CHANNELS_PER_GROUP = 128;
int choose_algo = -1;

cudnnMathType_t MATH_TYPE = CUDNN_DEFAULT_MATH;

cudnnTensorFormat_t INPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
int INPUT_BATCH_SIZE = 16;
int INPUT_CHANNELS = 0;
int INPUT_HEIGHT = 112;
int INPUT_WIDTH = 112;

cudnnTensorFormat_t OUTPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
int OUTPUT_BATCH_SIZE = 0, OUTPUT_CHANNELS = 0, OUTPUT_HEIGHT = 0,
    OUTPUT_WIDTH = 0;

cudnnTensorFormat_t KERNEL_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
int KERNEL_OUT_CHANNELS = 256; // #kernels = #output.channels
int KERNEL_HEIGHT = 3;
int KERNEL_WIDTH = 3;

int PAD_HEIGHT = 0;
int PAD_WIDTH = 0;
int VERTICAL_STRIDE = 1;
int HORIZONTAL_STRIDE = 1;
int DILATION_HEIGHT = 1;
int DILATION_WIDTH = 1;
#define CONV_MODE CUDNN_CROSS_CORRELATION

cudnnConvolutionFwdAlgo_t CONV_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#define CONV_ALGO_PREFER CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
#define MEMORY_LIMIT 0

int ROUNDS = 10;

namespace ch {
using namespace std::chrono;
}

const char algo_name[8][50] = {
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
};

const char math_types[3][50] = {
    "CUDNN_DEFAULT_MATH",
    "CUDNN_TENSOR_OP_MATH",
    "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",
};

//const cudnnConvolutionFwdAlgo_t total_conv_algo[] = {
    //CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    //CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    //CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    //CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    //CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    //CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    //CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    //CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
//};

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define checkCUDNN(expression)                                                 \
    {                                                                          \
        cudnnStatus_t status(expression);                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "Error on line " << __LINE__ << ": "                  \
                      << cudnnGetErrorString(status) << std::endl;             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }

double get_durtime(struct timeval t1, struct timeval t2);

bool isdigit(const char *str) {
    for (int i = 0; i < strlen(str); ++i)
        if (str[i] < '0' || str[i] > '9')
            return false;
    return true;
}

void get_args(int argc, const char *argv[]) {
    int pos = 1;
    while (pos < argc) {
        if (pos + 1 < argc && !strcmp(argv[pos], "-g") &&
            isdigit(argv[pos + 1])) {
            GROUP_COUNT = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-ca") &&
            isdigit(argv[pos + 1])) {
            if (atoi(argv[pos + 1]) < 8)
                choose_algo = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        /*if (pos+1 < argc && !strcmp(argv[pos], "-cdt") &&
        isdigit(argv[pos+1])) { if (atoi(argv[pos+1]) == 16) choose_data_type =
        16; if (atoi(argv[pos+1]) == 32) choose_data_type = 32; pos += 2;
        continue;
        }*/

        if (pos + 1 < argc && !strcmp(argv[pos], "-n") &&
            isdigit(argv[pos + 1])) {
            INPUT_BATCH_SIZE = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-c") &&
            isdigit(argv[pos + 1])) {
            INPUT_CHANNELS_PER_GROUP = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-f") &&
            isdigit(argv[pos + 1])) {
            KERNEL_OUT_CHANNELS = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-insize") &&
            isdigit(argv[pos + 1])) {
            INPUT_HEIGHT = INPUT_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-h") &&
            isdigit(argv[pos + 1])) {
            INPUT_HEIGHT = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-w") &&
            isdigit(argv[pos + 1])) {
            INPUT_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }

        if (pos + 1 < argc && !strcmp(argv[pos], "-kers") &&
            isdigit(argv[pos + 1])) {
            KERNEL_HEIGHT = KERNEL_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-r") &&
            isdigit(argv[pos + 1])) {
            KERNEL_HEIGHT = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-s") &&
            isdigit(argv[pos + 1])) {
            KERNEL_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-pad") &&
            isdigit(argv[pos + 1])) {
            PAD_HEIGHT = atoi(argv[pos + 1]);
            PAD_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-ph") &&
            isdigit(argv[pos + 1])) {
            PAD_HEIGHT = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-pw") &&
            isdigit(argv[pos + 1])) {
            PAD_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-rounds") &&
            isdigit(argv[pos + 1])) {
            ROUNDS = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }

        if (!strcmp(argv[pos], "-nhwc")) {
            INPUT_TENSOR_FORMAT = OUTPUT_TENSOR_FORMAT = KERNEL_TENSOR_FORMAT =
                CUDNN_TENSOR_NHWC;
            pos += 1;
            continue;
        }
        if (!strcmp(argv[pos], "-nchw")) {
            INPUT_TENSOR_FORMAT = OUTPUT_TENSOR_FORMAT = KERNEL_TENSOR_FORMAT =
                CUDNN_TENSOR_NCHW;
            pos += 1;
            continue;
        }
        if (!strcmp(argv[pos], "-default-math")) {
            MATH_TYPE = CUDNN_DEFAULT_MATH;
            pos += 1;
            continue;
        }
        if (!strcmp(argv[pos], "-tensor-op-math")) {
            MATH_TYPE = CUDNN_TENSOR_OP_MATH;
            pos += 1;
            continue;
        }
        if (!strcmp(argv[pos], "-tensor-op-math-allow-conversion")) {
            MATH_TYPE = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
            pos += 1;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-dh") &&
            isdigit(argv[pos + 1])) {
            DILATION_HEIGHT = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-dw") &&
            isdigit(argv[pos + 1])) {
            DILATION_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-dilate") &&
            isdigit(argv[pos + 1])) {
            DILATION_HEIGHT = atoi(argv[pos + 1]);
            DILATION_WIDTH = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-sh") &&
            isdigit(argv[pos + 1])) {
            VERTICAL_STRIDE = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-sw") &&
            isdigit(argv[pos + 1])) {
            HORIZONTAL_STRIDE = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        if (pos + 1 < argc && !strcmp(argv[pos], "-stride") &&
            isdigit(argv[pos + 1])) {
            VERTICAL_STRIDE = atoi(argv[pos + 1]);
            HORIZONTAL_STRIDE = atoi(argv[pos + 1]);
            pos += 2;
            continue;
        }
        pos += 1;
    }
    INPUT_CHANNELS = INPUT_CHANNELS_PER_GROUP * GROUP_COUNT;
}

int main(int argc, const char *argv[]) {

    get_args(argc, argv);

    int KERNEL_IN_CHANNELS = INPUT_CHANNELS_PER_GROUP;
    struct timeval t1, t2;
    struct timeval total_t1, total_t2;
    double time_memcpy_htod = 0.0, time_memcpy_dtoh = 0.0;
    double time_conv = 0.0;

    cudaSetDevice(0);
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    gettimeofday(&total_t1, 0);

    int expr = ROUNDS;

    // input
    d_type *c_input;
    unsigned int input_size =
        INPUT_BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    size_t input_bytes = INPUT_BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT *
                         INPUT_WIDTH * sizeof(d_type);
    c_input = (d_type *)malloc(input_bytes);
    srand((unsigned)time(0));
    for (int j = 0; j < input_size; ++j)
        c_input[j] = (d_type)rand() / RAND_MAX;

    d_type *d_input{nullptr};
    cudaMalloc(&d_input, input_bytes);

    gettimeofday(&t1, 0);
    cudaMemcpy(d_input, c_input, input_bytes, cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaDeviceSynchronize());
    gettimeofday(&t2, 0);
    time_memcpy_htod += get_durtime(t1, t2);

    // kernel
    d_type *c_kernel;
    unsigned int kernel_size =
        KERNEL_OUT_CHANNELS * INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH;
    size_t kernel_bytes = KERNEL_OUT_CHANNELS * INPUT_CHANNELS * KERNEL_HEIGHT *
                          KERNEL_WIDTH * sizeof(d_type);
    c_kernel = (d_type *)malloc(kernel_bytes);
    for (int j = 0; j < kernel_size; ++j)
        c_kernel[j] = (d_type)rand() / RAND_MAX;

    d_type *d_kernel{nullptr};
    cudaMalloc(&d_kernel, kernel_bytes);

    gettimeofday(&t1, 0);
    cudaMemcpy(d_kernel, c_kernel, kernel_bytes, cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaDeviceSynchronize());
    gettimeofday(&t2, 0);
    time_memcpy_htod += get_durtime(t1, t2);

    gettimeofday(&t1, 0);

    // descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/INPUT_TENSOR_FORMAT,
                                          /*dataType=*/DATA_TYPE,
                                          /*batch_size=*/INPUT_BATCH_SIZE,
                                          /*channels=*/INPUT_CHANNELS,
                                          /*height=*/INPUT_HEIGHT,
                                          /*width=*/INPUT_WIDTH));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/DATA_TYPE,
                                          /*format=*/KERNEL_TENSOR_FORMAT,
                                          /*out_channels=*/KERNEL_OUT_CHANNELS,
                                          /*in_channels=*/KERNEL_IN_CHANNELS,
                                          /*kernel_height=*/KERNEL_HEIGHT,
                                          /*kernel_width=*/KERNEL_WIDTH));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(
        cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/PAD_HEIGHT,
                                        /*pad_width=*/PAD_WIDTH,
                                        /*vertical_stride=*/VERTICAL_STRIDE,
                                        /*horizontal_stride=*/HORIZONTAL_STRIDE,
                                        /*dilation_height=*/DILATION_HEIGHT,
                                        /*dilation_width=*/DILATION_WIDTH,
                                        /*mode=*/CONV_MODE,
                                        /*conputeType=*/DATA_TYPE));

    if (GROUP_COUNT > 1)
        checkCUDNN(cudnnSetConvolutionGroupCount(convolution_descriptor,
                                                 /*group_count=*/GROUP_COUNT));
    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, MATH_TYPE));
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convolution_descriptor, input_descriptor, kernel_descriptor,
        &OUTPUT_BATCH_SIZE, &OUTPUT_CHANNELS, &OUTPUT_HEIGHT, &OUTPUT_WIDTH));

    std::cout << "Rounds: " << expr << std::endl;
    std::cout << "Group count: " << GROUP_COUNT << ", "
              << "Math type: " << math_types[MATH_TYPE] << std::endl;
    std::cout << "Input dims: " << INPUT_BATCH_SIZE << ", " << INPUT_CHANNELS
              << ", " << INPUT_HEIGHT << ", " << INPUT_WIDTH << std::endl;
    std::cout << "Kernel dims: " << KERNEL_IN_CHANNELS << ", "
              << KERNEL_OUT_CHANNELS << ", " << KERNEL_HEIGHT << ", "
              << KERNEL_WIDTH << std::endl;
    std::cout << "Output dims: " << OUTPUT_BATCH_SIZE << ", " << OUTPUT_CHANNELS
              << ", " << OUTPUT_HEIGHT << ", " << OUTPUT_WIDTH << std::endl;

    size_t output_bytes = OUTPUT_BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_HEIGHT *
                          OUTPUT_WIDTH * sizeof(d_type);
    d_type *d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/OUTPUT_TENSOR_FORMAT,
                                          /*dataType=*/DATA_TYPE,
                                          /*batch_size=*/OUTPUT_BATCH_SIZE,
                                          /*channels=*/OUTPUT_CHANNELS,
                                          /*height=*/OUTPUT_HEIGHT,
                                          /*width=*/OUTPUT_WIDTH));
    // warmup for two times
    //for (int kk = 0; kk < 3; ++kk) {
        float best_time = 10000;
        //cudnnConvolutionFwdAlgo_t convolution_algorithm;
        //if (choose_algo == -1) {
            //// checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
            //// input_descriptor,
            //// kernel_descriptor,
            //// convolution_descriptor,
            //// output_descriptor,
            //// CONV_ALGO_PREFER,
            ////[>memoryLimitInByptes=<]MEMORY_LIMIT,
            ////&convolution_algorithm));
            //const int convAlgoCnt = 8;
            //// get default workspace
            //size_t workspaceSize = (size_t)1024 * 1024 * 1024;
            //float *workspace;
            //HANDLE_ERROR(cudaMalloc(&workspace, workspaceSize));
            //int cnt = 0;
            //cudnnConvolutionFwdAlgoPerf_t perfResults[convAlgoCnt];
            //checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
                //cudnn, input_descriptor, d_input, kernel_descriptor, d_kernel,
                //convolution_descriptor, output_descriptor, d_output,
                //convAlgoCnt, &cnt, perfResults, workspace, workspaceSize));
            //assert(cnt > 0);
            //checkCUDNN(perfResults[0].status);
            //convolution_algorithm = perfResults[0].algo;
            //// for (int i = 0; i < cnt; ++i) {
            ////     std::cout << "algo name: " << algo_name[perfResults[i].algo]
            ////               << ", time: " << perfResults[i].time << std::endl;
            //// }
            //HANDLE_ERROR(cudaDeviceSynchronize());
        //} else
            //convolution_algorithm = total_conv_algo[choose_algo];

        //std::cout << "Chosen algorithm: " << algo_name[convolution_algorithm]
                  //<< std::endl;

        for (int algoid = 0; algoid < 8; ++algoid) {
            size_t workspace_bytes = 0;
            auto status((cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
                output_descriptor, (cudnnConvolutionFwdAlgo_t)algoid, &workspace_bytes)));
            if (status != CUDNN_STATUS_SUCCESS)
                continue;
            //std::cout << "Workspace size: " << (workspace_bytes) << "B"
                    //<< std::endl;

            void *d_workspace{nullptr};
            if (workspace_bytes > 0)
                cudaMalloc(&d_workspace, workspace_bytes);

            const float alpha = 1, beta = 0;
            ch::time_point<ch::high_resolution_clock, ch::nanoseconds> beg, end;
            time_conv = 0;
            int warmup = 200;
            for (int i = 0; i < warmup + expr; ++i) {
                HANDLE_ERROR(cudaDeviceSynchronize());
                beg = ch::high_resolution_clock::now();
                auto status(cudnnConvolutionForward(
                    cudnn, &alpha, input_descriptor, d_input, kernel_descriptor,
                    d_kernel, convolution_descriptor, 
                    //convolution_algorithm,
                    (cudnnConvolutionFwdAlgo_t)algoid,
                    d_workspace, workspace_bytes, &beta, output_descriptor,
                    d_output));
                if (status != CUDNN_STATUS_SUCCESS)
                    break;
                HANDLE_ERROR(cudaDeviceSynchronize());
                end = ch::high_resolution_clock::now();
                if (i >= warmup)
                    time_conv +=
                        ch::duration_cast<ch::duration<double>>(end - beg).count() *
                        1000;
            }
	    time_conv /= expr;
            std::cout << algo_name[algoid] << ": " << time_conv  << std::endl;
            if (time_conv < best_time)
                best_time = time_conv;
            cudaFree(d_workspace);
        }
    //}
        std::cout << "best time: " << best_time << std::endl;

    d_type *h_output = new d_type[output_bytes];

    gettimeofday(&t1, 0);
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    HANDLE_ERROR(cudaDeviceSynchronize());
    gettimeofday(&t2, 0);
    time_memcpy_dtoh += get_durtime(t1, t2);

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);

    HANDLE_ERROR(cudaDeviceSynchronize());
    gettimeofday(&total_t2, 0);
    //time_total = get_durtime(total_t1, total_t2);

    //printf("TFlops: %.2lf tflops\n",
           //2.0 * INPUT_BATCH_SIZE * INPUT_CHANNELS_PER_GROUP * OUTPUT_HEIGHT *
               //OUTPUT_WIDTH * KERNEL_OUT_CHANNELS * KERNEL_HEIGHT *
               //KERNEL_WIDTH / VERTICAL_STRIDE / HORIZONTAL_STRIDE / 1e9 /
               //time_conv * expr);
    //printf("memcpy_htod: %.6lf ms, memcpy_dtoh: %.6lf ms\n",
           //time_memcpy_htod / expr, time_memcpy_dtoh / expr);
    //printf("choose: %.6lf ms, convolution: %.6lf ms, convtotal: %.6lf ms, "
           //"total: %.6lf ms\n",
           //time_choose / expr, time_conv / expr, time_conv_total / expr,
           //time_total / expr);

    return 0;
}

double get_durtime(struct timeval t1, struct timeval t2) {
    return (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) /
           1000.0;
}

