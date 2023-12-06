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

template
class perceptron::activation::Linear<Scalar>;

template
class perceptron::activation::Sigmoid<Scalar>;

template
class perceptron::activation::ReLU<Scalar>;

template
class perceptron::activation::TanH<Scalar>;
