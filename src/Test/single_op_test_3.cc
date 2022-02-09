#include "code_engine.h"
#include "graph.h"
#include "operator.h"
#include "search_engine.h"
#include "tensor.h"

int main() {
    auto g = new tpm::Graph();
    auto a = g->tensor({1, 2048, 2048});
    auto b = g->tensor({1, 2048, 2048});
    auto c = g->tensor({1, 2048, 2048});

    g->matmul(a, b, c);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine<tpm::Generator> searchEngine;
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");

    return 0;
}
