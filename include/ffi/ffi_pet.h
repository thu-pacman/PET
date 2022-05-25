#pragma once
#include <pybind11/stl.h>
namespace py = pybind11;

namespace tpm {

void initCodeEngine(py::module &);
void initSearchEngine(py::module &m);
void initGraph(py::module &m);

} // namespace tpm