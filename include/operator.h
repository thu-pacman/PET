#ifndef OPERATOR_H
#define OPERATOR_H

#include "common.h"
#include "perf.h"
#include "tensor.h"
#include "transpose.h"
#include <functional>
#include <sstream>

namespace tpm {

class PerfEngine;

class Operator {
  public:
    enum OpType {
        Unknown = 0,
        // linear
        Conv = 100,
        Matmul,
        Pad,
        Slice,
        Concat,
        Split,
        Transpose,
        Extend,
        MaxPool,
        AvgPool,
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Gather,
        ReduceMean,
        Reshape,
        Identity,
        // element wise
        BatchNorm = 200,
        Softmax,
        Activation,
    };

    enum ActType {
        None,
        Relu,
        Sigmoid,
    };

    // TODO : remove all generateHash in ops;
    Operator() : guid(generateGuid()), hash(generateHash()) {}
    Operator(OpType type)
        : guid(generateGuid()), hash(generateHash()), type(type) {}
    Operator(OpType type, TensorVec inputs, TensorVec outputs)
        : guid(generateGuid()), hash(generateHash()), type(type),
          inputs(inputs), outputs(outputs) {}
    Operator(OpType type, TensorVec inputs)
        : guid(generateGuid()), hash(generateHash()), type(type),
          inputs(inputs) {}
    Operator(const Operator &rhs)
        : guid(generateGuid()), hash(rhs.hash), type(rhs.type) {}

    virtual ~Operator() {}

    bool isLinearOp() const { return type >= 100 && type < 200; }
    bool isElementWiseOp() const { return type >= 200; }
    bool isSplitOp() const { return type == Split; }
    bool isConcatOp() const { return type == Concat; }
    bool isComputeOp() const { return type == Conv || type == Matmul; }
    bool isTransposeOp() const { return type == Transpose; }

    // if operator does not have any connection with other ops or tensors
    bool isClear() {
        return inputs.empty() && outputs.empty() && successors.empty() &&
               predecessors.empty();
    }
    // clear all the connection of this operator
    void clear() {
        inputs.clear();
        outputs.clear();
        successors.clear();
        predecessors.clear();
    }

    void setSuccessors(OpVec suc) { successors = suc; }
    void setPredecessors(OpVec pre) { predecessors = pre; }

    void addSuccessors(Operator *suc) { successors.emplace_back(suc); }
    void addPredecessors(Operator *pre) { predecessors.emplace_back(pre); }

    OpVec &getSuccessors() { return successors; }
    const OpVec &getSuccessors() const { return successors; }
    OpVec &getPredecessors() { return predecessors; }
    const OpVec &getPredecessors() const { return predecessors; }
    Operator *getPredecessor() {
        return predecessors.size() != 1 ? nullptr : predecessors[0];
    }

    TensorVec &getInputs() { return inputs; }
    const TensorVec &getInputs() const { return inputs; }
    TensorVec &getOutputs() { return outputs; }
    const TensorVec &getOutputs() const { return outputs; }
    void setInputs(const std::vector<Tensor *> &inputs_) { inputs = inputs_; }
    void setOutputs(const std::vector<Tensor *> &outputs_) {
        outputs = outputs_;
    }
    Tensor *getInputs(size_t i) {
        return inputs.size() > i ? inputs[i] : nullptr;
    }
    Tensor *getOutput() const { return isSplitOp() ? nullptr : outputs[0]; }

    void addInput(Tensor *input) { inputs.emplace_back(input); }
    void addOutput(Tensor *output) { outputs.emplace_back(output); }

    virtual double perf(PerfEngine *pe, int rounds = 200,
                        int warmupRounds = 200) = 0;

    // guid will be the same after clone
    // predecessors and successors are not cloned
    virtual Operator *clone() = 0;

    /**
     * Compute part of the operator
     * @param dr : Destination points
     * @return : pair(input points, function to compute it)
     *           The function returning true means success
     */
    virtual std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) = 0;

    virtual Tensor *compute() = 0;
    virtual bool compute(const TensorVec &inputs, const TensorVec &outputs) {
        if (computeShape(inputs, outputs) && compute() != nullptr)
            return true;
        return false;
    }

  protected:
    virtual bool checkValid(const TensorVec &inputs) = 0;
    virtual void initHash() = 0;

  public:
    virtual Dim computeShape() = 0;
    virtual bool computeShape(const TensorVec &inputs,
                              const TensorVec &outputs) {
        if ((type != Concat && inputs.size() != (size_t)numInputs()) ||
            (type != Split && outputs.size() != (size_t)numOutputs()))
            return false;
        for (auto &&item : inputs)
            if (item == nullptr)
                return false;
        for (auto &&item : outputs)
            if (item == nullptr)
                return false;
        if (!checkValid(inputs))
            return false;
        this->inputs = inputs;
        this->outputs = outputs;
        auto dm = computeShape();
        for (auto output : outputs)
            if (!output->isValid())
                return false;
        outputs[0]->setDims(dm);
        return true;
    }

    virtual int numInputs() = 0;

    virtual int numOutputs() = 0;

    virtual void print() const final { std::cout << toString() << std::flush; }
    virtual std::string toString() const = 0;

    size_t getGuid() const { return guid; }

    uint64_t getHash() const { return hash; }

    OpType getType() const { return type; }

    virtual void inferSplittingPoints() {
        fprintf(stderr, "This op has unrealized inferSplittingPoints.\n");
        assert(false);
    }

    virtual void
    getOptypeAttr(std::string &optype, std::map<std::string, std::string> &attr,
                  std::map<std::string, std::vector<int>> &extra) const = 0;

  protected:
    const size_t guid;
    uint64_t hash;
    OpType type;
    TensorVec inputs;
    TensorVec outputs;
    OpVec predecessors;
    OpVec successors;
};

class ConvOp : public Operator {
  public:
    // When PaddingMode is Other, ConvOp will use padding size (ph, pw)
    // Otherwise, padding size (ph, pw) will be computed by padding mode
    enum PaddingMode {
        Other,
        Same,
        Valid,
    };

  private:
    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

    // Padding mode is set at the constructor which only set padding size
    void setPaddingMode();
    // Padding size is set when computeShape() is called
    // Actually we only need to call this function when the inputs are reset,
    // i.e., computeShape(TensorVec, TensorVec) is called. However,
    // computeShape(TensorVec, TensorVec) is only implemented as a virtual
    // function in base class Operator, so we call setPaddingSize in
    // computeShape()
    // Implicitly called in computeShape() now
    // void setPaddingSize();

  public:
    // Constructors for explicitly setting padding size
    ConvOp(Tensor *input, Tensor *weight, Tensor *output, int ph, int pw,
           int sh = 1, int sw = 1, int dh = 1, int dw = 1,
           Tensor *bias = nullptr, ActType act = None);
    ConvOp(Tensor *input, Tensor *weight, int ph, int pw, int sh = 1,
           int sw = 1, int dh = 1, int dw = 1, Tensor *bias = nullptr,
           ActType act = None);
    ConvOp(int ph, int pw, int sh = 1, int sw = 1, int dh = 1, int dw = 1,
           Tensor *bias = nullptr, ActType act = None);
    // Constructors for setting padding mode
    ConvOp(Tensor *input, Tensor *weight, Tensor *output, PaddingMode pm = Same,
           int sh = 1, int sw = 1, int dh = 1, int dw = 1,
           Tensor *bias = nullptr, ActType act = None);
    ConvOp(Tensor *input, Tensor *weight, PaddingMode pm = Same, int sh = 1,
           int sw = 1, int dh = 1, int dw = 1, Tensor *bias = nullptr,
           ActType act = None);
    ConvOp(PaddingMode pm = Same, int sh = 1, int sw = 1, int dh = 1,
           int dw = 1, Tensor *bias = nullptr, ActType act = None);
    ConvOp(const ConvOp &rhs);

    ConvOp *clone() override { return new ConvOp(*this); }

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Tensor *compute() override;

    Dim computeShape() override;

    Dim computeOutputPenalty(const Dim &p);

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override;

    std::string toString() const override;
    int numInputs() override { return 2; }
    int numOutputs() override { return 1; }

    Tensor *getBias() const { return bias; }

    void setAct(ActType act) { this->act = act; }
    ActType getAct() const { return act; }

    void inferSplittingPoints() override;

    bool same(const ConvOp &rhs);

    PaddingMode getPaddingMode() { return padding; }

    int getDh() const { return dh; }
    int getDw() const { return dw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

    ConvArgs getArgs(int withPenalty) const {
        auto input = inputs[0], weight = inputs[1];
        auto n = input->getDims()[0] + withPenalty * input->getPenalty()[0];
        auto c = input->getDims()[1] + withPenalty * input->getPenalty()[1];
        auto h = input->getDims()[2] + withPenalty * input->getPenalty()[2];
        auto w = input->getDims()[3] + withPenalty * input->getPenalty()[3];
        auto f = weight->getDims()[0];
        auto cpg = weight->getDims()[1];
        auto r = weight->getDims()[2];
        auto s = weight->getDims()[3];
        auto g = c / cpg;
        auto bi = (bias == nullptr) ? 0 : 1;
        auto ac = act;
        return ConvArgs{n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, g, bi, ac};
    }

  private:
    int ph, pw;
    int sh, sw;
    int dh, dw;
    Tensor *bias; // not part of the graph connections
    ActType act;
    PaddingMode padding;
};

class MatmulOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

    void checkAndSetTensorTypeForConstructor(Tensor *A, Tensor *B);

  public:
    MatmulOp(Tensor *A, Tensor *B, Tensor *C, bool transA = false,
             bool transB = false, Tensor *bias = nullptr, ActType act = None)
        : Operator(Matmul, {A, B}, {C}), transA(transA), transB(transB),
          bias(bias), act(act) {
        checkAndSetTensorTypeForConstructor(A, B);
        assert(checkValid({A, B}));
        initHash();
    }

    MatmulOp(Tensor *A, Tensor *B, bool transA = false, bool transB = false,
             Tensor *bias = nullptr, ActType act = None);

    MatmulOp(bool transA = false, bool transB = false, Tensor *bias = nullptr,
             ActType act = None);

    MatmulOp(const MatmulOp &rhs)
        : Operator(rhs), transA(rhs.transA), transB(rhs.transB), bias(rhs.bias),
          act(rhs.act) {}

    MatmulOp *clone() override { return new MatmulOp(*this); }

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Tensor *compute() override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override;

    std::string toString() const override {
        std::ostringstream os;
        MatmulArgs args = getArgs();
        os << "Matmul([" << (transA ? "A^T" : "A") << ","
           << (transB ? "B^T" : "B") << ",act=" << (int)act
           << "],A=" << inputs[0]->getHash() << ",B=" << inputs[1]->getHash()
           << ",C=" << outputs[0]->getHash()
           << ", TTbmnk: " << std::get<0>(args) << ", " << std::get<1>(args)
           << ", " << std::get<2>(args) << ", " << std::get<3>(args) << ", "
           << std::get<4>(args) << ", " << std::get<5>(args) << ")";
        return os.str();
    }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

    int numInputs() override { return 2; }
    int numOutputs() override { return 1; }

    Tensor *getBias() const { return bias; }

    void setAct(ActType act) { this->act = act; }
    ActType getAct() const { return act; }

    void inferSplittingPoints() override;

    bool getTransA() const { return transA; }
    bool getTransB() const { return transB; }

    MatmulArgs getArgs() const {
        auto A = inputs[0], B = inputs[1];
        auto b = A->getDims()[0];
        auto m = transA ? A->getDims()[2] : A->getDims()[1];
        auto n = transB ? B->getDims()[1] : B->getDims()[2];
        auto k = transA ? A->getDims()[1] : A->getDims()[2];
        return MatmulArgs{transA, transB, b, m, n, k};
    }

    void dimExtend(Tensor *t);

  private:
    bool transA, transB;
    Tensor *bias; // not part of the graph connections
    ActType act;
};

// Pad zeros at the begin and/or the end of each dimension
class PadOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

  public:
    PadOp(Tensor *input, Tensor *output, const Dim &begin, const Dim &end);
    PadOp(Tensor *input, const Dim &begin, const Dim &end);
    PadOp(const Dim &begin, const Dim &end);
    PadOp(const PadOp &rhs);

    PadOp *clone() override { return new PadOp(*this); }

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Tensor *compute() override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override;

    std::string toString() const override;
    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }

    const Dim &getBegin() const { return begin; }
    const Dim &getEnd() const { return end; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    Dim begin, end;
};

// Just like a[begin0:-end0, begin1:-end1]
class SliceOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

  public:
    SliceOp(Tensor *input, Tensor *output, const Dim &begin, const Dim &end);
    SliceOp(Tensor *input, const Dim &begin, const Dim &end);
    SliceOp(const Dim &begin, const Dim &end);
    SliceOp(const SliceOp &rhs);

    SliceOp *clone() override { return new SliceOp(*this); }

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Tensor *compute() override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override;

    std::string toString() const override;
    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }

    const Dim &getBegin() const { return begin; }
    const Dim &getEnd() const { return end; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    Dim begin, end;
};

class ConcatOp : public Operator {
  private:
    bool checkValid() {
        auto output = outputs[0];
        auto outputDims = output->getDims();
        int concatDim = 0;
        for (auto input : inputs) {
            assert(input != nullptr);
            auto inputDims = input->getDims();
            if (inputDims.size() != outputDims.size())
                return false;
            for (size_t i = 0; i < inputDims.size(); ++i)
                if (i != (size_t)dim) {
                    if (inputDims[i] != outputDims[i])
                        return false;
                    else
                        concatDim += inputDims[i];
                }
        }
        return concatDim == outputDims[dim];
    }

    bool checkValid(const TensorVec &tensors) override;
    void initHash() override;

  public:
    ConcatOp(TensorVec inputs, Tensor *output, int dim)
        : Operator(Concat, inputs, {output}), dim(dim) {
        checkValid();
        initHash();
        if (output->getDims().size() == 0)
            computeShape();
        assert(output != nullptr && output->getDims().size() > (size_t)dim);
    }

    ConcatOp(TensorVec inputs, int dim);

    ConcatOp(int dim);

    ConcatOp(const ConcatOp &rhs) : Operator(rhs), dim(rhs.dim) {}

    ConcatOp *clone() override { return new ConcatOp(*this); }

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Tensor *compute() override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Concat(dim=" << dim << ",i1=" << inputs[0]->getHash()
           << ",i2=" << inputs[1]->getHash() << ",out=" << outputs[0]->getHash()
           << ", dims = [" << inputs[0]->getDims()[dim] << ", "
           << inputs[1]->getDims()[dim] << "]"
           << ")";
        return os.str();
    }

    // TODO: concatting more than 2 tensors?
    int numInputs() override { return 2; }
    int numOutputs() override { return 1; }

    void inferSplittingPoints() override;

    int getDim() const { return dim; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int dim;
};

class SplitOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

    void setSizesForEqualySplit() {
        auto input = inputs[0];
        if (!sizes.empty())
            return;
        sizes = std::vector<int>(num, input->getDims()[dim] / num);
        // The last tensor after splitting will be smaller than others if the
        // dim size cannot be devided by num
        if (input->getDims()[dim] % num != 0)
            sizes[sizes.size() - 1] = input->getDims()[dim] % num;
    }

  public:
    SplitOp(Tensor *input, TensorVec outputs, int dim, int num)
        : Operator(Split, {input}, outputs), dim(dim), num(num), sizes({}) {
        assert(input != nullptr && !outputs.empty());
        initHash();
    }

    SplitOp(Tensor *input, TensorVec outputs, int dim,
            const std::vector<int> &sizes)
        : Operator(Split, {input}, outputs), dim(dim), num(-1), sizes(sizes) {
        assert(input != nullptr && !outputs.empty());
        assert(outputs.size() == sizes.size());
        initHash();
        computeShapeV();
    }

    // TODO: split with array param
    SplitOp(Tensor *input, int dim, int num);

    SplitOp(Tensor *input, int dim, const std::vector<int> &sizes);

    SplitOp(int dim, int num);

    SplitOp(int dim, const std::vector<int> &sizes);

    SplitOp(const SplitOp &rhs)
        : Operator(rhs), dim(rhs.dim), num(rhs.num), sizes(rhs.sizes) {}

    SplitOp *clone() override { return new SplitOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(size_t idx, DimRange dr);

    TensorVec computeV();
    bool compute(const TensorVec &inputTensors,
                 const TensorVec &outputTensors) override;

    Dim computeShape() override;
    bool computeShape(const TensorVec &inputTensors,
                      const TensorVec &outputTensors) override;

    std::vector<Dim> computeShapeV();

    const std::vector<int> &getSizes() { return sizes; }

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Split(dim=" << dim << ",num=" << num
           << ",in=" << inputs[0]->getHash() << ",o1=" << outputs[0]->getHash()
           << ",o2=" << outputs[1]->getHash() << ", dims = ["
           << outputs[0]->getDims()[dim] << ", " << outputs[1]->getDims()[dim]
           << "]"
           << ")";
        return os.str();
    }

    // TODO: splitting num > 2?
    int numInputs() override { return 1; }
    int numOutputs() override { return 2; }

    int getDim() const { return dim; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int dim, num;
    std::vector<int> sizes;
};

class PermItem {
    friend class Perm;
    std::vector<int> items;

  public:
    PermItem(int i) : items({i}) {}
    PermItem(std::initializer_list<int> lst) : items(lst) {}
    PermItem(const std::vector<int> &lst) : items(lst) {}

    bool isSingle() const { return items.size() == 1; }

    int getSingle() const { return items[0]; }

    std::vector<int> &getVec() { return items; }
    const std::vector<int> &getVec() const { return items; }

    std::string toString() const {
        std::string ret;
        if (isSingle())
            ret.append(std::to_string(items[0]));
        else {
            ret.append("{");
            for (auto i : items) {
                ret.append(std::to_string(i));
                ret.append(",");
            }
            ret.pop_back();
            ret.append("}");
        }
        return ret;
    }
};

class Perm {
    std::vector<PermItem> perm;

  public:
    Perm(std::initializer_list<PermItem> perm) : perm(perm) {}
    Perm(const std::vector<PermItem> &perm) : perm(perm) {}

    std::vector<int> asVector() const {
        std::vector<int> ret;
        for (auto &pi : perm)
            for (auto i : pi.items)
                ret.emplace_back(i);
        return ret;
    }

    std::vector<PermItem> &getPerm() { return perm; }

    PermItem &operator[](size_t idx) { return perm[idx]; }
    const PermItem &operator[](size_t idx) const { return perm[idx]; }

    size_t size() const { return perm.size(); }

    std::string toString() const {
        std::string ret;
        ret.append("{");
        for (auto &pi : perm) {
            ret.append(pi.toString());
            ret.append(",");
        }
        ret.pop_back();
        ret.append("}");
        return ret;
    }
};

class TransposeOp : public Operator {
  public:
    enum TransType {
        NoneType,
        N2H,
        N2W,
        H2N,
        W2N,
        C2H,
        C2W,
        D2H,
        D2W,
    };

    enum TransPos {
        NonePos,
        Pre,
        Post,
    };

  private:
    Perm before;
    Perm after;

    void setParam(int dimSz) {
        int idx = 0;
        for (int i = 0; i < dimSz; ++i) {
            if (i == split)
                before.getPerm().emplace_back(PermItem({idx++, idx++}));
            else
                before.getPerm().emplace_back(idx++);
        }
        if (split >= 0) {
            // Change the index of splitting introduced dim from -1 to split+1
            for (size_t i = 0, iEnd = after.size(); i < iEnd; ++i) {
                auto &pm = after[i].getVec();
                for (auto &item : pm) {
                    if (item > split)
                        item += 1;
                    if (item == -1)
                        item = split + 1;
                }
            }
        }
    }

    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

  public:
    TransposeOp(Tensor *input, Tensor *output, const Perm &before,
                const Perm &after, int factor = 2,
                TransType trans_type = NoneType);
    // The index of the splitting introduced dim is -1
    TransposeOp(Tensor *input, Tensor *output, int split, const Perm &after,
                int factor = 2, TransType trans_type = NoneType);
    TransposeOp(Tensor *input, int split, const Perm &after, int factor = 2,
                TransType trans_type = NoneType);
    TransposeOp(int split, const Perm &after, int factor = 2,
                TransType trans_type = NoneType);
    TransposeOp(const TransposeOp &rhs);

    TransposeOp *clone() override { return new TransposeOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override;

    std::vector<std::shared_ptr<TransBasic>> getTTParam() const;

    std::string toString() const override;
    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }
    void inferSplittingPoints() override;

    void setPos(TransPos pos) { trans_pos = pos; }
    TransPos getPos() const { return trans_pos; }
    void setType(TransType type) { trans_type = type; }
    TransType getType() const { return trans_type; }
    std::pair<int, int> getPaddingSize() const {
        return {padding_h, padding_w};
    }
    void setPaddingSize(int h, int w) {
        padding_h = h;
        padding_w = w;
    }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int split, factor;
    TransType trans_type;
    TransPos trans_pos = NonePos;
    int padding_h = 0, padding_w = 0;
};

class ExtendOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    ExtendOp(Tensor *input, Tensor *output, int dim, int num = 1)
        : Operator(Extend, {input}, {output}), dim(dim), num(num) {
        assert(input != nullptr && output != nullptr);
        initHash();
    }

    ExtendOp(Tensor *input, int dim, int num);

    ExtendOp(int dim, int num);

    ExtendOp(const ExtendOp &rhs) : Operator(rhs), dim(rhs.dim), num(rhs.num) {}

    ExtendOp *clone() override { return new ExtendOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Extend(dim=" << dim << ",num=" << num
           << ",in=" << inputs[0]->getHash() << ",out=" << outputs[0]->getHash()
           << ")";
        return os.str();
    }

    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }

    int getDim() const { return dim; }
    int getNum() const { return num; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int dim, num;
};

class BatchNormOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

  public:
    BatchNormOp(Tensor *input, Tensor *scale, Tensor *bias, Tensor *mean,
                Tensor *var, Tensor *output, float epsilon = 1e-05,
                float momentum = 0.9);

    BatchNormOp(Tensor *input, Tensor *scale, Tensor *bias, Tensor *mean,
                Tensor *var, float epsilon = 1e-05, float momentum = 0.9);

    BatchNormOp(Tensor *scale, Tensor *bias, Tensor *mean, Tensor *var,
                float epsilon = 1e-05, float momentum = 0.9)
        : Operator(BatchNorm), epsilon(epsilon), momentum(momentum),
          scale(scale), bias(bias), mean(mean), var(var) {
        initHash();
    }

    BatchNormOp(const BatchNormOp &rhs)
        : Operator(rhs), epsilon(rhs.epsilon), momentum(rhs.momentum),
          scale(rhs.scale->clone()), bias(rhs.bias->clone()),
          mean(rhs.mean->clone()), var(rhs.var->clone()) {
        initHash();
    }

    BatchNormOp *clone() override { return new BatchNormOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "BatchNorm(in=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ")";
        return os.str();
    }

    int numInputs() override { return 1; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    float epsilon, momentum;
    Tensor *scale, *bias, *mean, *var;
};

class MaxPoolOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    MaxPoolOp(Tensor *input, Tensor *output, int kh, int kw, int dh, int dw,
              int ph, int pw, int sh, int sw)
        : Operator(MaxPool, {input}, {output}), kh(kh), kw(kw), dh(dh), dw(dw),
          ph(ph), pw(pw), sh(sh), sw(sw) {
        initHash();
        if (output->getDims().size() == 0)
            computeShape();
    }

    MaxPoolOp(Tensor *input, int kh, int kw, int dh, int dw, int ph, int pw,
              int sh, int sw);

    MaxPoolOp(int kh, int kw, int dh, int dw, int ph, int pw, int sh, int sw);

    MaxPoolOp(const MaxPoolOp &rhs)
        : Operator(rhs), kh(rhs.kh), kw(rhs.kw), dh(rhs.dh), dw(rhs.dw),
          ph(rhs.ph), pw(rhs.pw), sh(rhs.sh), sw(rhs.sw) {}

    MaxPoolOp *clone() override { return new MaxPoolOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds = 200,
                int warmupRounds = 200) override;

    std::string toString() const override {
        std::ostringstream os;
        os << "MaxPool(size=[" << kh << "," << kw << "],p=[" << ph << "," << pw
           << "],s=[" << sh << "," << sw << "],d=[" << dh << "," << dw
           << "],in=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ")";
        return os.str();
    }

    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }

    int getKh() const { return kh; }
    int getKw() const { return kw; }
    int getDh() const { return dh; }
    int getDw() const { return dw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int kh, kw;
    int dh, dw;
    int ph, pw;
    int sh, sw;
};

class AvgPoolOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    AvgPoolOp(Tensor *input, Tensor *output, int kh, int kw, int ph, int pw,
              int sh, int sw)
        : Operator(AvgPool, {input}, {output}), kh(kh), kw(kw), ph(ph), pw(pw),
          sh(sh), sw(sw) {
        initHash();
        assert(inputs[0]->getDims().size() >= 2);
        computeShape();
    }

    // GlobalAveragePool
    AvgPoolOp(Tensor *input, Tensor *output)
        : Operator(AvgPool, {input}, {output}), ph(0), pw(0), sh(1), sw(1) {
        assert(input->getDims().size() == 4);
        kh = input->getDims()[2];
        kw = input->getDims()[3];
        initHash();
        assert(inputs[0]->getDims().size() >= 2);
        computeShape();
    }

    AvgPoolOp(Tensor *input, int kh, int kw, int ph, int pw, int sh, int sw);

    AvgPoolOp(int kh, int kw, int ph, int pw, int sh, int sw);

    AvgPoolOp(const AvgPoolOp &rhs)
        : Operator(rhs), kh(rhs.kh), kw(rhs.kw), ph(rhs.ph), pw(rhs.pw),
          sh(rhs.sh), sw(rhs.sw) {}

    AvgPoolOp *clone() override { return new AvgPoolOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds = 200,
                int warmupRounds = 200) override;

    std::string toString() const override {
        std::ostringstream os;
        os << "AvgPool(size=[" << kh << "," << kw << "],p=[" << ph << "," << pw
           << "],s=[" << sh << "," << sw << "],in=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ")";
        return os.str();
    }

    int numInputs() override { return 1; }

    int numOutputs() override { return 1; }

    int getKh() const { return kh; }
    int getKw() const { return kw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int kh, kw;
    int ph, pw;
    int sh, sw;
};

class AddOp : public Operator {
  private:
    bool checkValid(const TensorVec &tensors) override;
    void initHash() override;

  public:
    AddOp(TensorVec inputs, Tensor *output) : Operator(Add, inputs, {output}) {
        assert(checkValid(inputs));
        initHash();
    }

    AddOp(TensorVec inputs);

    AddOp();

    AddOp(const AddOp &rhs) : Operator(rhs) {}

    AddOp *clone() override { return new AddOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Add(i1=" << inputs[0]->getHash()
           << ",i2=" << inputs[1]->getHash() << ",out=" << outputs[0]->getHash()
           << ")";
        return os.str();
    }

    int numInputs() override { return 2; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;
};

class SubOp : public Operator {
  private:
    bool checkValid(const TensorVec &tensors) override;
    void initHash() override;

  public:
    SubOp(TensorVec inputs, Tensor *output) : Operator(Sub, inputs, {output}) {
        assert(checkValid(inputs));
        initHash();
    }

    SubOp(TensorVec inputs);

    SubOp();

    SubOp(const SubOp &rhs) : Operator(rhs) {}

    SubOp *clone() override { return new SubOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override { return "Sub()"; }

    int numInputs() override { return 2; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;
};

class MulOp : public Operator {
  private:
    bool checkValid(const TensorVec &tensors) override;
    void initHash() override;

  public:
    MulOp(TensorVec inputs, Tensor *output) : Operator(Mul, inputs, {output}) {
        assert(checkValid(inputs));
        initHash();
    }

    MulOp(TensorVec inputs);

    MulOp();

    MulOp(const MulOp &rhs) : Operator(rhs) {}

    MulOp *clone() override { return new MulOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Mul(i1=" << inputs[0]->getHash()
           << ",i2=" << inputs[1]->getHash() << ",out=" << outputs[0]->getHash()
           << ")";
        return os.str();
    }

    int numInputs() override { return 2; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;
};

class DivOp : public Operator {
  private:
    bool checkValid(const TensorVec &tensors) override;
    void initHash() override;

  public:
    DivOp(TensorVec inputs, Tensor *output) : Operator(Div, inputs, {output}) {
        assert(checkValid(inputs));
        initHash();
    }

    DivOp(TensorVec inputs);

    DivOp();

    DivOp(const DivOp &rhs) : Operator(rhs) {}

    DivOp *clone() override { return new DivOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override { return "Div()"; }

    int numInputs() override { return 2; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;
};

class PowOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override;
    void initHash() override;

  public:
    PowOp(Tensor *input, Tensor *output, int pow)
        : Operator(Pow, inputs, {output}), pow(pow) {
        assert(checkValid(inputs));
        initHash();
    }

    PowOp(Tensor *input, int pow);

    PowOp(int pow);

    PowOp(const PowOp &rhs) : Operator(rhs), pow(rhs.pow) {}

    PowOp *clone() override { return new PowOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override { return "Div()"; }

    int numInputs() override { return 1; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int pow;
};

class GatherOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    GatherOp(Tensor *data, Tensor *indices, Tensor *output, int axis)
        : Operator(Gather, {data, indices}, {output}), axis(axis) {
        initHash();
    }

    GatherOp(Tensor *data, Tensor *indices, int axis);

    GatherOp(const GatherOp &rhs) : Operator(rhs), axis(rhs.axis) {}

    GatherOp() : Operator(Gather) { initHash(); }

    GatherOp *clone() override { return new GatherOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Gather(data=" << inputs[0]->getHash()
           << ",indices=" << inputs[1]->getHash()
           << ",out=" << outputs[0]->getHash() << ",axis=" << axis << ")";
        return os.str();
    }

    int numInputs() override { return 2; }
    int numOutputs() override { return 1; }

    int getAxis() const { return axis; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int axis;
};

class ReduceMeanOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override {
        int nDim = inputs[0]->getDims().size();
        return axis >= 0 && axis < nDim;
    }
    void initHash() override;

  public:
    ReduceMeanOp(Tensor *input, Tensor *output, int axis)
        : Operator(ReduceMean, {input}, {output}), axis(axis) {
        initHash();
    }

    ReduceMeanOp(Tensor *input, int axis);

    ReduceMeanOp(const ReduceMeanOp &rhs) : Operator(rhs), axis(rhs.axis) {}

    ReduceMeanOp() : Operator(ReduceMean) { initHash(); }

    ReduceMeanOp *clone() override { return new ReduceMeanOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "ReduceMean(input=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ",axis=" << axis << ")";
        return os.str();
    }

    int numInputs() override { return 1; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int axis;
};

// Reshape to whatever output
// The shape is not recorded in the op
class ReshapeOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    ReshapeOp(Tensor *input, Tensor *output)
        : Operator(Reshape, {input}, {output}) {
        initHash();
    }

    ReshapeOp(const ReshapeOp &rhs) : Operator(rhs) {}

    ReshapeOp() : Operator(Reshape) { initHash(); }

    ReshapeOp *clone() override { return new ReshapeOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override {
        assert(false);
        return {};
    }

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Reshape(in=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ")";
        return os.str();
    }

    int numInputs() override { return 1; }

    int numOutputs() override { return 1; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;
};

class IdentityOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    IdentityOp(Tensor *input, Tensor *output)
        : Operator(Identity, {input}, {output}) {
        initHash();
    }

    IdentityOp(Tensor *input);

    IdentityOp(const IdentityOp &rhs) : Operator(rhs) {}

    IdentityOp() : Operator(Identity) { initHash(); }

    IdentityOp *clone() override { return new IdentityOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Identity(in=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ")";
        return os.str();
    }

    int numInputs() override { return 1; }

    int numOutputs() override { return 1; }

    void inferSplittingPoints() override {
        outputs[0]->setSplittingPoints(*inputs[0]->getSplittingPoints());
    };

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;
};

class SoftmaxOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override {
        int nDim = inputs[0]->getDims().size();
        return axis >= 0 && axis < nDim;
    }
    void initHash() override;

  public:
    SoftmaxOp(Tensor *input, Tensor *output, int axis)
        : Operator(Softmax, {input}, {output}), axis(axis) {
        initHash();
    }

    SoftmaxOp(Tensor *input, int axis);

    SoftmaxOp(const SoftmaxOp &rhs) : Operator(rhs), axis(rhs.axis) {}

    SoftmaxOp() : Operator(Softmax) { initHash(); }

    SoftmaxOp *clone() override { return new SoftmaxOp(*this); }

    Tensor *compute() override;

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override;

    Dim computeShape() override;

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return inputs[0]->getDims()[0] * 0.02;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Softmax(input=" << inputs[0]->getHash()
           << ",out=" << outputs[0]->getHash() << ",axis=" << axis << ")";
        return os.str();
    }

    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }

    int getAxis() const { return axis; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    int axis;
};

class ActivationOp : public Operator {
  private:
    bool checkValid(const TensorVec &inputs) override { return true; }
    void initHash() override;

  public:
    ActivationOp(Tensor *input, Tensor *output, ActType actType)
        : Operator(Activation, {input}, {output}), actType(actType) {
        initHash();
        if (output->getDims().size() == 0)
            output->setDims(computeShape());
    }

    ActivationOp(Tensor *input, ActType actType)
        : Operator(Activation, {input}, {}), actType(actType) {
        auto tensor = new Tensor();
        outputs.emplace_back(tensor);
        tensor->setDims(computeShape());
        initHash();
    }

    ActivationOp(const ActivationOp &rhs)
        : Operator(rhs), actType(rhs.actType) {}

    ActivationOp() : Operator(Activation) { initHash(); }

    ActivationOp *clone() override { return new ActivationOp(*this); }

    Tensor *compute() override {
        assert(false);
        return nullptr;
    }

    std::pair<std::vector<DimRange>, std::function<bool()>>
    compute(DimRange dr) override {
        assert(false);
        return {{}, nullptr};
    }

    Dim computeShape() override {
        outputs[0]->setType(inputs[0]->getType());
        return inputs[0]->getDims();
    }

    double perf(PerfEngine *pe, int rounds, int warmupRounds) override {
        return 0.0;
    }

    std::string toString() const override {
        std::ostringstream os;
        os << "Activation(act=" << actType << ", input=" << inputs[0]->getHash()
           << ", output=" << outputs[0]->getHash() << ")";
        return os.str();
    }

    int numInputs() override { return 1; }
    int numOutputs() override { return 1; }

    ActType getActType() const { return actType; }

    void getOptypeAttr(
        std::string &optype, std::map<std::string, std::string> &attr,
        std::map<std::string, std::vector<int>> &extra) const override;

  private:
    ActType actType;
};

} // end of namespace tpm

#endif // OPERATOR_H