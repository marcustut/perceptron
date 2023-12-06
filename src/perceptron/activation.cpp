#include "perceptron/activation.hpp"

template<typename T>
T perceptron::activation::Linear<T>::apply(const T x) {
    return x;
}

template<typename T>
T perceptron::activation::Linear<T>::differentiate([[maybe_unused]] const T x) {
    return 1.0;
}

template<typename T>
T perceptron::activation::Sigmoid<T>::apply(const T x) {
    return 1.0 / (1.0 + std::exp(-x));
}

template<typename T>
T perceptron::activation::Sigmoid<T>::differentiate(const T x) {
    auto y = this->apply(x);
    return y * (1.0 - y);
}

template<typename T>
T perceptron::activation::ReLU<T>::apply(const T x) {
    return std::fmax(0.0, x);
}

template<typename T>
T perceptron::activation::ReLU<T>::differentiate(const T x) {
    return x <= 0 ? 0.0 : 1.0;
}

template<typename T>
T perceptron::activation::TanH<T>::apply(const T x) {
    return std::tanh(x);
}

template<typename T>
T perceptron::activation::TanH<T>::differentiate(const T x) {
    auto y = this->apply(x);
    return 1.0 - std::pow(y, 2);
}


template<typename T>
T perceptron::activation::Softmax<T>::apply(const T x) {
    throw std::runtime_error("Softmax activation function must be applied to a vector");
}

template<typename T>
T perceptron::activation::Softmax<T>::differentiate(const T x) {
    throw std::runtime_error("Softmax activation function must be applied to a vector");
}

template<typename T>
nc::NdArray<T> perceptron::activation::Softmax<T>::apply(const nc::NdArray<T> &_x) {
    auto x = _x.numCols() == 1 ? _x.transpose() : _x;
    auto max_x = nc::max(x, nc::Axis::COL).reshape(x.shape().rows, 1);
    auto e_x = nc::exp(x - max_x);
    auto a = e_x / nc::sum(e_x, nc::Axis::COL).reshape(e_x.shape().rows, 1);
    return a.transpose();
}

template<typename T>
nc::NdArray<T> perceptron::activation::Softmax<T>::differentiate(const nc::NdArray<T> &x) {
    auto s = this->apply(x);
    auto out = nc::NdArray<T>(x.numRows(), x.numCols());
    for (size_t i = 0; i < x.numRows(); i++)
        for (size_t j = 0; j < x.numCols(); j++)
            out(i, j) = s[i] * (i == j ? 1.0 - s[j] : -s[j]);
    return out;
}

template
class perceptron::activation::Linear<Scalar>;

template
class perceptron::activation::Sigmoid<Scalar>;

template
class perceptron::activation::ReLU<Scalar>;

template
class perceptron::activation::TanH<Scalar>;

template
class perceptron::activation::Softmax<Scalar>;
