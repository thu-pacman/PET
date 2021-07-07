# from CCore cimport Operator
# from CCore cimport Tensor
# from CCore cimport Graph
from CCore cimport *
import ctypes

cdef class PyTensor:
    cdef Tensor* ctensor

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.ctensor = <Tensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.ctensor = <Tensor*>(ptr)

    property tensor:
        def __get__(self):
            if self.ctensor == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.ctensor, ctypes.c_void_p)

        def __set__(self, value):
            self._set_tensor(value)

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

cdef class PyGraph:
    cdef Graph *cgraph

    def __cinit__(self):
        self.cgraph = new Graph()

    def conv(self, guid, PyTensor input, PyTensor weight,
             PyTensor output, int ph, int pw, int sh = 1, int sw = 1,
             int dh = 1, int dw = 1, int g = 1):
        self.cgraph.conv(guid, input.ctensor, weight.ctensor, output.ctensor, ph, pw, sh, sw, dh, dw, g)

    def matmul(self, guid, PyTensor A, PyTensor B, PyTensor C):
        self.cgraph.matmul(guid, A.ctensor, B.ctensor, C.ctensor)

    def concat(self, guid, list inputs, PyTensor output, dim):
        cdef vector[Tensor*] cinputs
        cdef unsigned long long ptr
        cinputs.resize(len(inputs))
        for i in range(len(inputs)):
            assert(type(inputs[i]) == PyTensor)
            assert(inputs[i].tensor is not None)
            ptr = ctypes.cast(inputs[i].tensor, ctypes.c_void_p).value
            cinputs[i] = <Tensor*>(ptr)
        self.cgraph.concat(guid, cinputs, output.ctensor, dim)

    def relu(self, guid, PyTensor input, PyTensor output):
        self.cgraph.relu(guid, input.ctensor, output.ctensor)

    def sigmoid(self, guid, PyTensor input, PyTensor output):
        self.cgraph.sigmoid(guid, input.ctensor, output.ctensor)

    def tensor(self, guid, tuple dims):
        cdef vector[int] cdims
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]
        cdef Tensor *tensor = self.cgraph.tensor(guid, cdims)
        t = ctypes.cast(<unsigned long long>tensor, ctypes.c_void_p)
        return PyTensor(t)
