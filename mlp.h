// mlp.h
// Aswin van Woudenberg

#ifndef MLP_H
#define MLP_H

#include <cstdlib>
#include <ctime>
#include <array>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>

template<std::size_t SIZE>
struct Node {
    double output;
    double threshold;
    double thresholdDiff;
    double signalError;
    std::array<double, SIZE> weights;
    std::array<double, SIZE> weightDiffs;
    constexpr std::size_t size() const { return SIZE; }

    Node() : weightDiffs({0}), thresholdDiff(0.0)
    {
        auto rnd_func = [](double &d) 
            { d = -1 + 2 * (rand() / static_cast<double>(RAND_MAX)); };
        rnd_func(threshold);
        std::for_each(weights.begin(), weights.end(), rnd_func); 
    }
};

template<>
struct Node<0> { // inputnode
    double output;
    constexpr std::size_t size() const { return 0; }
};

template<std::size_t SIZE>
std::ostream& operator<<(std::ostream &os, const Node<SIZE> &node)
{
    os << node.threshold << " ";
    for (double d: node.weights) os << d << " ";
    return os;
}

template<> std::ostream& operator<<(std::ostream &os, const Node<0> &node) { return os; }

template<std::size_t SIZE>
std::istream& operator>>(std::istream &is, Node<SIZE> &node)
{
    is >> node.threshold;
    for (double &d: node.weights) is >> d;
    return is;
}

template<> std::istream& operator>>(std::istream &is, Node<0> &node) { return is; }

template<class TUPLE, std::size_t N, std::size_t K = N>
struct NodesFiller {
    static void setNodesFromTuple(const TUPLE &t, std::array<Node<0>, N> &nodes) {
        NodesFiller<TUPLE, N, K-1>::setNodesFromTuple(t, nodes);
        nodes[K-1].output = std::get<K-1>(t);
    }
};

template<class TUPLE, std::size_t N>
struct NodesFiller<TUPLE, N, 1> {
    static void setNodesFromTuple(const TUPLE &t, std::array<Node<0>, N> &nodes) {
        nodes[0].output = std::get<0>(t);
    }
};

template<typename FIRST, typename... REST>
void setNodesFromTuple(const std::tuple<FIRST, REST...> &t, std::array<Node<0>, 1+sizeof...(REST)> &nodes) {
    NodesFiller<decltype(t), 1+sizeof...(REST)>::setNodesFromTuple(t, nodes);
}

template<std::size_t SIZE, typename NODETYPE=Node<0>, typename LAYERTYPE=void>
struct Layer {
    LAYERTYPE previousLayer;
    std::array<NODETYPE, SIZE> nodes;
    constexpr std::size_t size() const { return SIZE; }
    
    static double sigmoid(const double net) { return 1 / (1 + exp(-net)); }

    void ff()
    {
        double net;
        previousLayer.ff();
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            net = nodes[i].threshold;
            for (std::size_t j = 0; j < previousLayer.size(); ++j)
            {
                net = net + previousLayer.nodes[j].output * nodes[i].weights[j];
            }
            nodes[i].output = sigmoid(net);
        }
    }

    template<typename... TYPES> void setInput(const std::tuple<TYPES...> &tuple) { previousLayer.setInput(tuple); }
    void setInput(const std::vector<double> &vector) { previousLayer.setInput(vector); }
    void setInput(double dbl) { previousLayer.setInput(dbl); }

    template<typename... TYPES> void calcErr(const std::tuple<TYPES...> &tuple) 
    {
        std::array<Node<0>,SIZE> expectedOutput;
        setNodesFromTuple<TYPES...>(tuple, expectedOutput);

        for (std::size_t i = 0; i < SIZE; ++i)
            nodes[i].signalError = (expectedOutput[i].output - nodes[i].output) * nodes[i].output * (1 - nodes[i].output);
        previousLayer.calcErr(nodes);
    }

    void calcErr(const std::vector<double> &expectedOutput)
    {
        for (std::size_t i = 0; i < SIZE; ++i)
            nodes[i].signalError = (expectedOutput[i] - nodes[i].output) * nodes[i].output * (1 - nodes[i].output);
        previousLayer.calcErr(nodes);
    }

    void calcErr(double expectedOutput)
    {
        nodes[0].signalError = (expectedOutput - nodes[0].output) * nodes[0].output * (1 - nodes[0].output);
        previousLayer.calcErr(nodes);
    }

    template<std::size_t NEXTTYPE, std::size_t NEXTSIZE> 
    void calcErr(const std::array<Node<NEXTTYPE>,NEXTSIZE> &nextNodes)
    {
        for (std::size_t j = 0; j < SIZE; ++j) 
        {
            double sum = 0;
            for (std::size_t k = 0; k < NEXTSIZE; ++k)
                sum += nextNodes[k].weights[j] * nextNodes[k].signalError;
            nodes[j].signalError = nodes[j].output * (1 - nodes[j].output) * sum;
        }
        previousLayer.calcErr(nodes);
    }

    void bpErr(const double learningRate, const double momentum)
    {
        for (std::size_t j = 0; j < SIZE; ++j) 
        {
            nodes[j].thresholdDiff = learningRate * nodes[j].signalError + momentum * nodes[j].thresholdDiff;
            nodes[j].threshold += nodes[j].thresholdDiff;

            for (std::size_t k = 0; k < previousLayer.size(); ++k) 
            {
                nodes[j].weightDiffs[k] = learningRate * nodes[j].signalError * previousLayer.nodes[k].output + 
                                          momentum * nodes[j].weightDiffs[k];
                nodes[j].weights[k] = nodes[j].weights[k] + nodes[j].weightDiffs[k];
            }
        }
        previousLayer.bpErr(learningRate, momentum);
    }

};

template<std::size_t SIZE>
struct Layer<SIZE,Node<0>,void> { // inputlayer
    std::array<Node<0>, SIZE> nodes;
    constexpr std::size_t size() const { return SIZE; }
    void ff() { }
    template<class... TYPES> void setInput(const std::tuple<TYPES...> &tuple) { setNodesFromTuple<TYPES...>(tuple, nodes); }
    void setInput(const std::vector<double> &vector) { for (int i = 0; i<SIZE; ++i) nodes[i].output = vector[i]; }
    void setInput(double dbl) { nodes[0].output = dbl; }
    template<std::size_t NEXTTYPE, std::size_t NEXTSIZE> void calcErr(const std::array<Node<NEXTTYPE>,NEXTSIZE> &nextNodes) { }
    void bpErr(const double learningRate, const double momentum) { };
};

template<std::size_t SIZE, typename NODETYPE=Node<0>, typename LAYERTYPE=void>
std::ostream& operator<<(std::ostream &os, const Layer<SIZE,NODETYPE,LAYERTYPE> &layer)
{
    os << layer.previousLayer;
    for (auto &node: layer.nodes) os << node;
    return os;
}

template<std::size_t SIZE> std::ostream& operator<<(std::ostream &os, const Layer<SIZE,Node<0>,void> &layer) { return os; }

template<std::size_t SIZE, typename NODETYPE=Node<0>, typename LAYERTYPE=void>
std::istream& operator>>(std::istream &is, Layer<SIZE,NODETYPE,LAYERTYPE> &layer)
{
    is >> layer.previousLayer;
    for (auto &node: layer.nodes) is >> node;
    return is;
}

template<std::size_t SIZE> std::istream& operator>>(std::istream &is, Layer<SIZE,Node<0>,void> &layer) { return is; }

template<std::size_t I, std::size_t H1, std::size_t H2=0, std::size_t O=0>
struct Mlp { // 4 layer MLP
    Layer<O,Node<H2>,Layer<H2,Node<H1>,Layer<H1,Node<I>,Layer<I>>>> mlp;
    constexpr std::size_t size() const { return 4; }
};

template<std::size_t I, std::size_t H, std::size_t O>
struct Mlp<I,H,O,0> { // 3 layer MLP
    Layer<O,Node<H>,Layer<H,Node<I>,Layer<I>>> mlp;
    constexpr std::size_t size() const { return 3; }
};

template<std::size_t I, std::size_t O>
struct Mlp<I,O,0,0> { // 2 layer MLP
    Layer<O,Node<I>,Layer<I>> mlp;
    constexpr std::size_t size() const { return 2; }
};

template<std::size_t... SIZES>
std::ostream& operator<<(std::ostream &os, const Mlp<SIZES...> &mlp)
{ 
    os << mlp.mlp; 
    return os; 
}

template<std::size_t... SIZES>
std::istream& operator>>(std::istream &is, Mlp<SIZES...> &mlp)
{
    is >> mlp.mlp;
    return is;
}

template<std::size_t... SIZES>
struct Bpn : public Mlp<SIZES...> {
    double overallError;
    constexpr std::size_t size() const { return sizeof...(SIZES); }

    Bpn() { }
    Bpn(const std::string &filename) { load(filename); }

    template<class INPUT>
    void ff(const INPUT &input)
    {
        this->mlp.setInput(input);
        this->mlp.ff();
    }

    void save(const std::string &filename) const 
    {
        std::ofstream ofs(filename, std::ofstream::trunc);
        ofs << *this;
        ofs.close();
    }

    void load(const std::string &filename) 
    {
        std::ifstream ifs(filename);
        ifs >> *this;
        ifs.close();
    }

    template<typename... TYPES> double calcErr(const std::tuple<TYPES...> &tuple) const 
    {
        std::array<Node<0>,sizeof...(TYPES)> expectedOutput;
        setNodesFromTuple<TYPES...>(tuple, expectedOutput);

        double error = 0.0;
        for (std::size_t j = 0; j < this->mlp.size(); ++j)
            error += 0.5 * pow(expectedOutput[j].output - this->mlp.nodes[j].output, 2);
        return error;
    }

    double calcErr(const std::vector<double> &expectedOutput) const
    {
        double error = 0.0;
        for (std::size_t j = 0; j < this->mlp.size(); ++j)
            error += 0.5 * pow(expectedOutput[j] - this->mlp.nodes[j].output, 2);
        return error;
    }

    double calcErr(double expectedOutput) const
    {
        return 0.5 * pow(expectedOutput - this->mlp.nodes[0].output, 2);
    }

    template<class INPUT, class OUTPUT>
    void train(const std::vector<INPUT> &input, const std::vector<OUTPUT> &output, 
        const unsigned long maxIter = 2000, const double minError = 0.09,
        const double learningRate = 0.3, const double momentum = 0.9)
    {
        if (input.size() != output.size())
            throw std::invalid_argument("input and output vectors are of unequal length");

        unsigned long iter = 0;
        do {
            overallError = 0.0;
            for (std::size_t i = 0; i < input.size(); ++i) {
                ff(input[i]);
                overallError += calcErr(output[i]);
                this->mlp.calcErr(output[i]);
                this->mlp.bpErr(learningRate, momentum);
            }
            iter++;
        } while ((overallError > minError) && (iter < maxIter || maxIter == 0));
    }

    template<class INPUT, class OUTPUT>
    double test(const std::vector<INPUT> &input, const std::vector<OUTPUT> &output)
    {
        if (input.size() != output.size())
            throw std::invalid_argument("input and output vectors are of unequal length");

        overallError = 0.0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            ff(input[i]);
            overallError += calcErr(output[i]);
        }
        return overallError;
    }

};

#endif // MLP_H

