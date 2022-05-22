cmake_minimum_required(VERSION 3.9) # Required by find_package(OpenMP)
project(TPM C CXX)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG") # Still enable assertion
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG") # Still enable assertion

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++11 when on, -std=c++11 when off

find_package(PythonInterp)

add_subdirectory(3rd-party/pybind11)
add_subdirectory(3rd-party/nlohmann_json_cmake_fetchcontent)

include_directories(include)
include_directories(3rd-party/pybind11/include)
include_directories(3rd-party/nlohmann_json_cmake_fetchcontent/single_include)
file(GLOB_RECURSE SRC src/*.cc src/*.cu)
file(GLOB_RECURSE TESTS src/Test/*.cc)
file(GLOB_RECURSE FFIS src/ffi/*.cc)
list(REMOVE_ITEM SRC ${TESTS})
list(REMOVE_ITEM SRC ${FFIS})

# CUDA
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Target
cuda_add_library(tpm SHARED ${SRC})
cuda_add_cublas_to_target(tpm) # cublas
target_link_libraries(tpm cudnn curand)
target_link_libraries(tpm pybind11::embed)
target_link_libraries(tpm nlohmann_json::nlohmann_json)

# Python bindings
pybind11_add_module(cpp_module MODULE ${FFIS})
target_link_libraries(cpp_module PRIVATE tpm)

# Tests
set(BUILD_PET_TEST ON CACHE BOOL "Build test for PET")
set(BUILD_NNET_TEST ON CACHE BOOL "Build test for NNET (based on GTest)")

add_executable(onnx_pet src/Test/import_onnx.cpp)
target_link_libraries(onnx_pet tpm)

add_executable(onnx_nnet src/Test/import_onnx_nnet.cpp)
target_link_libraries(onnx_nnet tpm)

add_executable(onnx_cmutator src/Test/import_onnx_cmutator.cpp)
target_link_libraries(onnx_cmutator tpm)

add_executable(onnx_origin src/Test/import_onnx_origin.cpp)
target_link_libraries(onnx_origin tpm)

add_executable(onnx_depth src/Test/import_onnx_nnet_depth.cpp)
target_link_libraries(onnx_depth tpm)

if (BUILD_PET_TEST)
    add_executable(conv1 src/Test/conv_test_1.cc)
    target_link_libraries(conv1 tpm)

    add_executable(conv2 src/Test/conv_test_2.cc)
    target_link_libraries(conv2 tpm)

    add_executable(conv3 src/Test/conv_test_3.cc)
    target_link_libraries(conv3 tpm)

    add_executable(conv4 src/Test/conv_test_4_perf.cc)
    target_link_libraries(conv4 tpm)

    add_executable(conv5 src/Test/conv_test_5_trans_kernel.cc)
    target_link_libraries(conv5 tpm)

    add_executable(convtrans1 src/Test/convtrans_test_1.cc)
    target_link_libraries(convtrans1 tpm)

    add_executable(convtrans2 src/Test/convtrans_test_2.cc)
    target_link_libraries(convtrans2 tpm)

    add_executable(dilation src/Test/dilation_test.cpp)
    target_link_libraries(dilation tpm)

    add_executable(graph1 src/Test/SampleGraph1.cpp)
    target_link_libraries(graph1 tpm)

    add_executable(graph2 src/Test/SampleGraph2.cpp)
    target_link_libraries(graph2 tpm)

    add_executable(graph3 src/Test/SampleGraph3.cpp)
    target_link_libraries(graph3 tpm)

    add_executable(graph4 src/Test/SampleGraph4.cpp)
    target_link_libraries(graph4 tpm)

    add_executable(mutant1 src/Test/mutant_test_1.cc)
    target_link_libraries(mutant1 tpm)

    add_executable(mutant2 src/Test/mutant_test_2.cc)
    target_link_libraries(mutant2 tpm)

    add_executable(mutant3 src/Test/mutant_test_3.cc)
    target_link_libraries(mutant3 tpm)

    add_executable(trans1 src/Test/transpose_test_1.cc)
    target_link_libraries(trans1 tpm)

    add_executable(trans2 src/Test/transpose_test_2.cc)
    target_link_libraries(trans2 tpm)

    add_executable(trans3 src/Test/transpose_test_3.cc)
    target_link_libraries(trans3 tpm)

    add_executable(trans4 src/Test/transpose_test_4.cc)
    target_link_libraries(trans4 tpm)

    add_executable(trans5 src/Test/transpose_test_5.cc)
    target_link_libraries(trans5 tpm)

    add_executable(tensor src/Test/tensor_test.cc)
    target_link_libraries(tensor tpm)

    add_executable(rule1 src/Test/single_rule_test_1_n2h.cc)
    target_link_libraries(rule1 tpm)

    add_executable(rule2 src/Test/single_rule_test_2_n2w.cc)
    target_link_libraries(rule2 tpm)

    add_executable(rule3 src/Test/single_rule_test_3_c2h.cc)
    target_link_libraries(rule3 tpm)

    add_executable(rule4 src/Test/single_rule_test_4_c2w.cc)
    target_link_libraries(rule4 tpm)

    add_executable(rule5 src/Test/single_rule_test_5_dilated.cc)
    target_link_libraries(rule5 tpm)

    add_executable(rule6 src/Test/single_rule_test_6_bgemm.cc)
    target_link_libraries(rule6 tpm)

    add_executable(rule7 src/Test/single_rule_test_7_trans_gemm.cc)
    target_link_libraries(rule7 tpm)

    add_executable(rule8 src/Test/single_rule_test_8_conv2group.cc)
    target_link_libraries(rule8 tpm)

    add_executable(rule9 src/Test/single_rule_test_9_group_conv_a.cc)
    target_link_libraries(rule9 tpm)

    add_executable(rule10 src/Test/single_rule_test_10_group_conv_b.cc)
    target_link_libraries(rule10 tpm)

    add_executable(rule11 src/Test/single_rule_test_11_group_conv_c.cc)
    target_link_libraries(rule11 tpm)

    add_executable(rule12 src/Test/single_rule_test_12_1xk_trans.cc)
    target_link_libraries(rule12 tpm)

    add_executable(rule13 src/Test/single_rule_test_13_nx1xk_trans.cc)
    target_link_libraries(rule13 tpm)

    add_executable(rule14 src/Test/single_rule_test_14_split_group_conv.cc)
    target_link_libraries(rule14 tpm)

    add_executable(rule15 src/Test/single_rule_test_15_conv_1x1.cc)
    target_link_libraries(rule15 tpm)

    add_executable(rule16 src/Test/single_rule_test_16_group_conv_d.cc)
    target_link_libraries(rule16 tpm)

    add_executable(op1 src/Test/single_op_test_1.cc)
    target_link_libraries(op1 tpm)

    add_executable(op2 src/Test/single_op_test_2.cc)
    target_link_libraries(op2 tpm)

    add_executable(op3 src/Test/single_op_test_3.cc)
    target_link_libraries(op3 tpm)

    add_executable(op4 src/Test/single_op_test_4.cc)
    target_link_libraries(op4 tpm)

    add_executable(op5 src/Test/single_op_test_5.cc)
    target_link_libraries(op5 tpm)

    add_executable(op6 src/Test/single_op_test_6.cc)
    target_link_libraries(op6 tpm)

    add_executable(op7 src/Test/single_op_test_7_batch_gemm.cc)
    target_link_libraries(op7 tpm)

    add_executable(te1 src/Test/eliminator_test_1.cc)
    target_link_libraries(te1 tpm)

    add_executable(split1 src/Test/split_test_1.cc)
    target_link_libraries(split1 tpm)

    add_executable(split2 src/Test/split_test_2.cc)
    target_link_libraries(split2 tpm)

    add_executable(split3 src/Test/split_test_3.cc)
    target_link_libraries(split3 tpm)

    add_executable(concat1 src/Test/concat_test_1.cc)
    target_link_libraries(concat1 tpm)

    add_executable(extend1 src/Test/extend_test_1.cc)
    target_link_libraries(extend1 tpm)

    add_executable(group1 src/Test/group_conv_test_1.cc)
    target_link_libraries(group1 tpm)

    add_executable(group2 src/Test/group_conv_test_2.cc)
    target_link_libraries(group2 tpm)

    add_executable(fuse_activation src/Test/fuse_activation.cpp)
    target_link_libraries(fuse_activation tpm)

    add_executable(gen_transpose src/Test/gen_transpose.cpp)
    target_link_libraries(gen_transpose tpm)

    add_executable(stat_graph1 src/Test/graph_stat_test_1_transpose_group.cc)
    target_link_libraries(stat_graph1 tpm)

    add_executable(stat_graph2 src/Test/graph_stat_test_2_batchmm.cc)
    target_link_libraries(stat_graph2 tpm)

    add_executable(pad_slice src/Test/pad_slice_test.cc)
    target_link_libraries(pad_slice tpm)

    add_executable(bgemm src/Test/batch_gemm_test.cc)
    target_link_libraries(bgemm tpm)

    add_executable(gemm1 src/Test/gemm_test_1_perf.cc)
    target_link_libraries(gemm1 tpm)

    add_executable(perf_pool src/Test/perf_pool.cc)
    target_link_libraries(perf_pool tpm)

    add_executable(cache1 src/Test/cache_test_1.cc)
    target_link_libraries(cache1 tpm)

    add_executable(cache2 src/Test/cache_test_2.cc)
    target_link_libraries(cache2 tpm)

    add_executable(cache3 src/Test/cache_test_3.cc)
    target_link_libraries(cache3 tpm)
endif()