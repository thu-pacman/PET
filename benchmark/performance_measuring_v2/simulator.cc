#include "simulator.h"
#include "operator.h"

void Simulator::initOp(std::vector<OpConfig> configVector) {
    for (auto config : configVector) {
        
        if (pfMap.find(config) != pfMap.end()) {
            pfMap[config] = PerformanceInfo();
            
            OpType opType = config.opType;
            if (opType == OpType::CONV2D) {
                // create a Conv2d operator
                opVector.push_back(dynamic_cast<Operator*>(new Conv2d(config)));
            }
            else if (opType == OpType::MATMUL) {
                // create a MatMul operator
				opVector.push_back(dynamic_cast<Operator*>(new MatMul(config)));
            }
        }
        
    }
}

void Simulator::freeOp() {
    for (auto op : opVector) {
        delete op;
    }
    opVector.clear();
}

PfMap Simulator::measureAllOp(int rounds) {
    
    // test the performance of operators and update pfMap
    for (auto op : opVector) {
        op->performance_measuring(this, rounds);
    }
    
    return pfMap;
}

// Operator call the following function to update the result
void Simulator::updatePfMap(OpConfig config, double durtime, double tflops, std::vector<int> otherInfo) {
    pfMap[config].update(durtime, tflops);
    for (auto info : otherInfo) {
       pfMap[config].setOtherInfo(info);
    }
}

