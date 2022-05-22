#include "ffi/ffi.h"
#include "ffi/ffi_pet.h"

// void start_interpreter() {
//     static py::scoped_interpreter *interpretor = nullptr;
//     if (interpretor == nullptr) {
//         interpretor = new py::scoped_interpreter(); // guard
//     }
// }

PYBIND11_MODULE(cpp_module, m) {
    tpm::initGraph(m);
    tpm::initCodeEngine(m);
    tpm::initSearchEngine(m);
}
