from libcpp.vector cimport vector

cdef extern from 'operator.h' namespace 'tpm':
    cdef enum OpType 'Operator::OpType':
        Unknown = 0,
        # linear
        Conv = 100,
        Matmul,
        Concat,
        # element wise & none linear
        Relu = 200,
        Sigmoid,
        # others
        # input, output & weight
        Input = 400,
        Output,
        Weight,

    cdef cppclass Operator:
        pass
    
    cdef cppclass Tensor:
        pass


cdef extern from 'graph.h' namespace 'tpm':
    cdef cppclass Graph:
        Graph() except +
        void updateConnection()
        Operator* conv(size_t guid, Tensor* input, Tensor* weight, 
                       Tensor* output, int ph, int pw, int sh,
                       int sw, int dh, int dw, int g)
        Operator* matmul(size_t guid, Tensor* A, Tensor* B, Tensor* C)
        Operator* concat(size_t guid, vector[Tensor*] inputs, 
                         Tensor *output, int dim)
        Operator* relu(size_t guid, Tensor* input, Tensor* output)
        Operator* sigmoid(size_t guid, Tensor* input, Tensor* output)
        Tensor* tensor(size_t guid, vector[int] dims)

