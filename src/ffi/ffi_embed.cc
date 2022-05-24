#include "ffi/ffi_embed.h"
namespace py = pybind11;

namespace tpm {
void start_interpreter() {
    static py::scoped_interpreter *interpretor = nullptr;
    if (interpretor == nullptr) {
        interpretor = new py::scoped_interpreter(); // guard
    }
}
} // end of namespace tpm
