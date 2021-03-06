#include "generator.h"
#include "graph.h"
#include "operator.h"
#include "search_engine.h"
#include "tensor.h"
#include <cstdlib>
#include <iostream>
#include <sys/time.h>

double getDurtime(struct timeval beg, struct timeval end) {
    double t =
        (1000000.0 * (end.tv_sec - beg.tv_sec) + end.tv_usec - beg.tv_usec) /
        1000.0;
    return t;
}

int main() {
    auto i0 = new tpm::Tensor({1, 512, 14, 14});
    auto i1 = new tpm::Tensor({1, 32, 12, 12});
    auto w0 = new tpm::Tensor({32, 512, 3, 3});
    auto op0 = new tpm::ConvOp(i0, w0, i1, 0, 0, 1, 1, 1, 1);

    i0->dataRand();
    i1->dataMalloc();
    w0->dataRand();

    struct timeval beg, end;
    gettimeofday(&beg, 0);
    op0->compute();
    gettimeofday(&end, 0);
    std::cout << "conv time: " << getDurtime(beg, end) << std::endl;

    delete i0;
    delete i1;
    delete w0;
    delete op0;

    return 0;
}
