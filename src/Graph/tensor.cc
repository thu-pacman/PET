#include "tensor.h"
#include "common.h"
#include "operator.h"

namespace tpm {
std::pair<Operator *, int> Tensor::getOutputOfWithIndex() {
    if (outputOf == nullptr)
        return {nullptr, -1};
    auto it = std::find(outputOf->getOutputs().begin(),
                        outputOf->getOutputs().end(), this);
    if (it != outputOf->getOutputs().end())
        return {outputOf, std::distance(it, outputOf->getOutputs().begin())};
    return {nullptr, -1};
}

bool Tensor::random_inited;
int Tensor::random_seed[256 * 16];

} // end of namespace tpm