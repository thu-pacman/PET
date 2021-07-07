#include "ffi.h"

namespace tpm {

void start_interpreter() {
    static py::scoped_interpreter *interpretor = nullptr;
    if (interpretor == nullptr) {
        interpretor = new py::scoped_interpreter(); // guard
    }
}

} // namespace tpm

