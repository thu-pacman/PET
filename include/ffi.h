#ifndef FFI_H
#define FFI_H

#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace tpm {

namespace py = pybind11;

void start_interpreter();

} // namespace tpm

#endif // FFI_H
