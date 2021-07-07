#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>
#include <map>
#include <iostream>

// Types of supported operators
enum OpType {
    CONV2D,
    MATMUL
};

// Configration of an operator
class OpConfig {
public:
    OpType opType;
    std::vector<int> args;

	OpConfig() {}
	
	bool operator<(const OpConfig other) {
		if (opType < other.opType) return true;

		if (opType > other.opType) return false;

		for (int i = 0; i < args.size(); ++ i) {
			if (args[i] < other.args[i]) { return true; }
			if (args[i] > other.args[i]) { return false;}
		}

		return false;
	}

	bool operator==(const OpConfig other) {
		if (opType == other.opType) {
			for (int i = 0; i < args.size(); ++ i) {
				if (args[i] != other.args[i]) { return false; }
			}
			return true;
		}

		return false;
	}

	OpConfig& operator=(OpConfig other) {
		opType = other.opType;
		args = other.args;
	}

	OpConfig(const OpConfig& other) {
		opType = other.opType;
		args = other.args;
	}
};

class PerformanceInfo {
public:
    PerformanceInfo(): durtime(0.0), tflops(0.0) {}
    PerformanceInfo(double durtime, double tflops):
        durtime(durtime), tflops(tflops) {}
    
    void update(double durtime, double tflops) {
        durtime = durtime;
        tflops = tflops;
    }
    
    void setOtherInfo(int info) { otherInfo.push_back(info); }
    
    double getDurtime() { return durtime; }
    double getTflops() { return tflops; }
    int getOtherInfoAt(int index) {
        if (index >= otherInfo.size()) {
            std::cerr << "index out of range!\n";
            exit(0);
        }
        return otherInfo[index];
    }
    
private:
    double durtime, tflops;
    std::vector<int> otherInfo;
};

class KeyCmp {
public:
	bool operator()(const OpConfig& cf1, const OpConfig& cf2) {
		if (cf1.opType < cf2.opType) return true;
		if (cf1.opType > cf2.opType) return false;

		// cf1.opType == cf2.opType
		for (int i = 0; i < cf1.args.size(); ++ i) {
			if (cf1.args[i] < cf2.args[i]) return true;
			if (cf1.args[i] > cf2.args[i]) return false;
		}

		return false;
	}
};

using PfMap = std::map<OpConfig, PerformanceInfo, KeyCmp>;
class Operator;

class Simulator {
    
    PfMap pfMap;
    std::vector<Operator*> opVector;
    
public:
    Simulator() {}
    
    // Initialize operators from the list of OpConfig
    void initOp(std::vector<OpConfig> configVector);
    void freeOp();
    
    PfMap measureAllOp(int rounds);
    void updatePfMap(OpConfig config, double durtime, double tflops, std::vector<int> othorInfo);
    
};

#endif
