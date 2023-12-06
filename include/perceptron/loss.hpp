#ifndef PERCEPTRON_LOSS_H
#define PERCEPTRON_LOSS_H

#include <vector>

#include "perceptron/blas.hpp"
#include "NumCpp.hpp"

using Matrix = perceptron::blas::Matrix;
using Blas = perceptron::blas::Blas;

namespace perceptron::loss {
    /**
     * The interface to be implemented by different loss functions
     */
    template<typename T>
    class Loss {
    public:
        virtual T apply(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const = 0;

        virtual nc::NdArray <T>
        differentiate(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const = 0;
    };

    /**
     * Mean Squared Error (MSE).
     */
    template<typename T>
    class MeanSquaredError : public Loss<T> {
    public:
        // MSE = sum((targets - predictions)^2) / n
        T apply(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const;

        // MSE' = -2 (targets - predictions) / n
        nc::NdArray <T> differentiate(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const;
    };

    /**
     * Sum of Squared Residuals (SSR).
     */
    template<typename T>
    class SSR : public Loss<T> {
    public:
        // SSR = sum((targets - predictions)^2)
        T apply(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const;

        // SSR' = -2 (targets - predictions)
        nc::NdArray <T> differentiate(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const;
    };

    /**
     * Cross Entropy Cost.
     */
    template<typename T>
    class CrossEntropy : public Loss<T> {
    public:
        // SSR = sum((targets - predictions)^2) / n
        T apply(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const;

        // SSR' = -2 (targets - predictions) / n
        nc::NdArray <T> differentiate(const nc::NdArray <T> &predictions, const nc::NdArray <T> &targets) const;
    };
} // namespace perceptron::loss

#endif //PERCEPTRON_LOSS_H
