#include "code_engine.h"
#include "common.h"
#include "ffi/ffi.h"
#include "graph.h"
#include "operator.h"
#include "perf_engine.h"
#include "search_engine.h"
#include "tensor.h"

namespace tpm {

using namespace py::literals;

void initGraph(py::module &m) {
    py::class_<Tensor>(m, "Tensor");
    py::class_<Operator>(m, "Operator");
    py::class_<OpVec>(m, "OpVec");
    py::class_<TensorVec>(m, "TensorVec");
    py::class_<PermItem>(m, "PermItem")
        .def(py::init<>())
        .def(py::init<const std::vector<int> &>())
        .def(py::init<std::initializer_list<int>>());
    py::class_<Perm>(m, "Perm")
        .def(py::init<const std::vector<PermItem> &>())
        .def(py::init<std::initializer_list<PermItem>>());
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("tensor",
             static_cast<Tensor *(Graph::*)(const Dim &, const std::string &)>(
                 &Graph::tensor),
             py::return_value_policy::reference_internal)
        .def("conv",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *, int,
                                              int, int, int, int, int,
                                              Tensor *)>(&Graph::conv),
             "input"_a, "weight"_a, "output"_a, "ph"_a, "pw"_a, "sh"_a = 1,
             "sw"_a = 1, "dh"_a = 1, "dw"_a = 1, "bias"_a = nullptr,
             py::return_value_policy::reference_internal)
        .def("convTrans",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *, int,
                                              int, int, int, int, int, int, int,
                                              Tensor *)>(&Graph::convTrans),
             py::return_value_policy::reference_internal)
        .def("pad",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, const Dim &,
                                              const Dim &)>(&Graph::pad),
             py::return_value_policy::reference_internal)
        .def(
            "concat",
            static_cast<Operator *(Graph::*)(const TensorVec &, Tensor *, int)>(
                &Graph::concat),
            py::return_value_policy::reference_internal)
        .def(
            "matmul",
            static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *, bool,
                                             bool, Tensor *)>(&Graph::matmul),
            py::return_value_policy::reference_internal)
        .def("g2bmm",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *, int,
                                              int, Tensor *)>(&Graph::g2bmm),
             py::return_value_policy::reference_internal)
        .def("gbmml",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *, int,
                                              Tensor *)>(&Graph::gbmml),
             py::return_value_policy::reference_internal)
        .def("split",
             static_cast<Operator *(Graph::*)(Tensor *, const TensorVec &, int,
                                              int)>(&Graph::split),
             py::return_value_policy::reference_internal)
        .def("transpose",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int,
                                              const Perm &, int)>(
                 &Graph::transpose),
             py::return_value_policy::reference_internal)
        .def("flatten",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int)>(
                 &Graph::flatten),
             py::return_value_policy::reference_internal)
        .def("maxpool",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int, int, int,
                                              int, int, int, int, int)>(
                 &Graph::maxpool),
             py::return_value_policy::reference_internal)
        .def("avgpool",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int, int, int,
                                              int, int, int)>(&Graph::avgpool),
             py::return_value_policy::reference_internal)
        .def("avgpool",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *)>(
                 &Graph::avgpool),
             py::return_value_policy::reference_internal)
        .def("add",
             static_cast<Operator *(Graph::*)(const TensorVec &, Tensor *)>(
                 &Graph::add),
             py::return_value_policy::reference_internal)
        .def("sub",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *)>(
                 &Graph::sub),
             py::return_value_policy::reference_internal)
        .def("mul",
             static_cast<Operator *(Graph::*)(const TensorVec &, Tensor *)>(
                 &Graph::mul),
             py::return_value_policy::reference_internal)
        .def("div",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *)>(
                 &Graph::div),
             py::return_value_policy::reference_internal)
        .def("pow",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int)>(
                 &Graph::pow),
             py::return_value_policy::reference_internal)
        .def("gather",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *,
                                              int)>(&Graph::gather),
             py::return_value_policy::reference_internal)
        .def("reduceMean",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int)>(
                 &Graph::reduceMean),
             py::return_value_policy::reference_internal)
        .def("softmax",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, int)>(
                 &Graph::softmax),
             py::return_value_policy::reference_internal)
        .def("reshape",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *)>(
                 &Graph::reshape),
             py::return_value_policy::reference_internal)
        .def("sigmoid",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *)>(
                 &Graph::sigmoid),
             py::return_value_policy::reference_internal)
        .def(
            "relu",
            static_cast<Operator *(Graph::*)(Tensor *, Tensor *)>(&Graph::relu),
            py::return_value_policy::reference_internal)
        .def(
            "tanh",
            static_cast<Operator *(Graph::*)(Tensor *, Tensor *)>(&Graph::tanh),
            py::return_value_policy::reference_internal)
        .def("batchnorm",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *,
                                              Tensor *, Tensor *, Tensor *,
                                              float, float)>(&Graph::batchnorm),
             py::return_value_policy::reference_internal)
        .def("identity",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *)>(
                 &Graph::identity),
             py::return_value_policy::reference_internal)
        .def("split",
             static_cast<Operator *(Graph::*)(Tensor *, const TensorVec &, int,
                                              std::vector<int>)>(&Graph::split),
             py::return_value_policy::reference_internal)
        .def("slice",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *,
                                              Tensor *)>(&Graph::slice),
             py::return_value_policy::reference_internal)
        .def("resize",
             static_cast<Operator *(Graph::*)(Tensor *, Tensor *, Tensor *)>(
                 &Graph::resize),
             py::return_value_policy::reference_internal)
        .def("setInputs",
             static_cast<void (Graph::*)(TensorVec)>(&Graph::setInputs))
        .def("setOutputs",
             static_cast<void (Graph::*)(TensorVec)>(&Graph::setOutputs))
        .def("getOperators",
             static_cast<OpVec &(Graph::*)()>(&Graph::getOperators),
             py::return_value_policy::reference_internal)
        .def("updateConnection",
             static_cast<void (Graph::*)()>(&Graph::updateConnection));
    py::class_<SubGraph, std::shared_ptr<SubGraph>>(m, "SubGraph")
        .def(py::init<>())
        .def(py::init<OpVec>());
}

void initCodeEngine(py::module &m) {
    py::class_<CodeEngine>(m, "CodeEngine")
        .def(py::init<>())
        .def("importPerfEngine",
             static_cast<void (CodeEngine::*)(std::shared_ptr<PerfEngine>)>(
                 &CodeEngine::importPerfEngine))
        .def("genCode", static_cast<int (CodeEngine::*)(
                            std::shared_ptr<SubGraph> &, const std::string &)>(
                            &CodeEngine::genCode));
}

void initSearchEngine(py::module &m) {
    py::class_<PerfEngine, std::shared_ptr<PerfEngine>>(m, "PerfEngine");

    py::class_<SearchEngine>(m, "SearchEngine")
        .def(py::init<const std::shared_ptr<Mutator> &>())
        // FIXME: remove this ctor which binds to Generator
        .def(py::init(
            []() { return SearchEngine(std::make_shared<Generator>()); }))
        .def("run",
             static_cast<int (SearchEngine::*)(
                 const std::shared_ptr<SubGraph> &graph,
                 std::shared_ptr<SubGraph> &bestGraph)>(&SearchEngine::run))
        .def("exportPerfEngine",
             static_cast<std::shared_ptr<PerfEngine> (SearchEngine::*)()>(
                 &SearchEngine::exportPerfEngine),
             py::return_value_policy::reference_internal);
}

} // namespace tpm