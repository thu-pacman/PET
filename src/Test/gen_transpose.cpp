#include "code_engine.h"
#include "graph.h"
#include <iostream>
#include <vector>

int main() {
    tpm::Graph g;
    auto t = g.tensor({16, 32, 32, 32});
    auto op = g.transpose(t, 1, {0, 1, 2, {-1, 3}}, 8);
    tpm::CodeEngine engine;
    std::string funcCode, invokeCode;
    std::tie(funcCode, invokeCode) =
        engine.genTranspose({(tpm::TransposeOp *)op}, "t0", "I", "O");
    std::cout << funcCode << std::endl
              << "============" << std::endl
              << invokeCode << std::endl;
    return 0;
}

