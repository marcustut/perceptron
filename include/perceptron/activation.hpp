#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cmath"

#include "perceptron/blas.hpp"
#include "NumCpp.hpp"

namespace perceptron::activation {
    /**
     * The interface to be implemented by different activation functions
     */
    template<typename T>
    class Activation {
    public:
        virtual T apply(T x) = 0;

        virtual nc::NdArray<T> apply(const nc::NdArray<T> &x) {
            auto out = x;
            std::transform(x.cbegin(), x.cend(), out.begin(), [this](T x) { return this->apply(x); });
            return out;
        }

        virtual T differentiate(T x) = 0;

        virtual nc::NdArray<T> differentiate(const nc::NdArray<T> &x) {
            auto out = x;
            std::transform(x.begin(), x.end(), out.begin(), [this](T x) { return this->differentiate(x); });
            return out;
        }
    };

    /**
     * Most common activation function which is kind of pointless but is convenient for testing.
     */
    template<typename T>
    class Linear : public Activation<T> {
    public:
        // f(x) = x
        T apply(const T x);

        // f'(x) = 1
        T differentiate([[maybe_unused]]const T x);
    };

    /**
     * "S"-shaped curve or sigmoid curve.
     * See: https://en.wikipedia.org/wiki/Sigmoid_function
     */
    template<typename T>
    class Sigmoid : public Activation<T> {
    public:
        // f(x) = 1 / (1 + e^(-x))
        T apply(const T x);

        // f'(x) = f(x)(1 - f(x))
        T differentiate(const T x);
    };

    /**
     * Rectified Linear Unit
     * See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
     */
    template<typename T>
    class ReLU : public Activation<T> {
    public:
        // f(x) = 0 for x < 0 and x for x >= 0
        T apply(const T x);

        // f'(x) = 0 for x < 0 and 1 for x >= 0
        T differentiate(const T x);
    };

    /**
     * Tangent Hyperbolic
     * See: https://www.baeldung.com/cs/sigmoid-vs-tanh-functions#tanh
     */
    template<typename T>
    class TanH : public Activation<T> {
    public:
        // f(x) = 2 / (1 + e^(-2x)) - 1
        T apply(const T x);

        // f'(x) = 1 - f(x)^2
        T differentiate(const T x);
    };
} // namespace perceptron::activation

#endif //ACTIVATION_H
