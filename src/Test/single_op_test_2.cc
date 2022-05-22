#include "code_engine.h"
#include "graph.h"
#include "operator.h"
#include "search_engine.h"
#include "tensor.h"

int main() {
    // conv7x7->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu
    auto g = new tpm::Graph();
    auto i8 = g->tensor({64, 256, 14, 14});
    auto i9 = g->tensor({64, 256, 14, 14});

    auto w9 = g->tensor({256, 256, 3, 3});

    g->conv(i8, w9, i9, 2, 2, 1, 1, 2, 2);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::Generator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");

    return 0;
}
