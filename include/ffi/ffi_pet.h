#pragma once
#include "ffi/ffi.h"

namespace tpm {

void initCodeEngine(py::module &);
void initSearchEngine(py::module &m);
void initGraph(py::module &m);

} // namespace tpm