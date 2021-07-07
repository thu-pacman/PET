// This file defines some operators in DNN

#ifndef OPERATOR_H
#define OPERATOR_H

#include "simulator.h"
#include "cudnn.h"
#include <string>

class Simulator;

class PerformanceInfo {
  double durtime;
  double tflops;

public:
  PerformanceInfo() : durtime(0.0), tflops(0.0) {}
  PerformanceInfo(double durtime, double tflops)
      : durtime(durtime), tflops(tflops) {}

  void update(double d, double t) {
    durtime = d;
    tflops = t;
  }

  // fields
  // std::string id;
};

// class MetaRule {
// public:
// 	using MetaSignature = std::string;
// 	MetaSignature signature;

// };

// // (n,c,h,w) -> (n/2,c,h*2,w)
// class ND2HM2 : MetaRule {
// 	ND2HM2() : signature("ND2HM2") {}
// };

// // (n,c,h,w) -> (n/2,c,h,w*2)
// class ND2WM2 : MetaRule {
// 	ND2WM2() : signature("ND2WM2") {}
// };

// // (n,c,h,w) -> (n*2,c,h/2,w)
// class NM2HD2 : MetaRule {
// 	NM2HD2() : signature("NM2HD2") {}
// };

// // (n,c,h,w) -> (n*2,c,h,w/2)
// class NM2WD2 : MetaRule {
// 	NM2WD2() : signature("NM2WD2") {}
// };

// class Signature {
// 	std::vector<MetaRule::MetaSignature> signature;
// };
using Signature = std::string;

class Operator {

protected:
  std::string id;

  PerformanceInfo perfInfo;

  Operator *baseOp;
  Operator *prevOp;
  Signature signature;

public:
  Operator() {}
  Operator(const std::string &_id) : id(_id) {}

  virtual void performance_measuring(Simulator *, int rounds) = 0;
};

// ---------------------------------------------------------------------------
// Conv2d op

class Conv2d : public Operator {

public:
  Conv2d() {}
  Conv2d(const std::string &id, int group, int batch_size, int in_channel,
         int out_channel, int in_h, int in_w, int kn_h, int kn_w, int pad_h,
         int pad_w, int stride_h, int stride_w, int dila_h, int dila_w);
  virtual void performance_measuring(Simulator *simu, int rounds);

private:
  int GROUP;
  int BATCH_SIZE, IN_CHANNEL, OUT_CHANNEL, IN_H, IN_W, KN_H, KN_W;
  int PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILA_H, DILA_W;
  cudnnConvolutionFwdAlgo_t algo;
};

class OperatorOptimizer {
  std::vector<Operator> candidates;
};

#endif
