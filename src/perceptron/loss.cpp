#include "perceptron/loss.hpp"

template<typename T>
T
perceptron::loss::MeanSquaredError<T>::apply(const nc::NdArray<T> &predictions,
                                             const nc::NdArray<T> &targets) const {
    return nc::mean(nc::square(targets - predictions))[0];
}

template<typename T>
nc::NdArray<T>
perceptron::loss::MeanSquaredError<T>::differentiate(const nc::NdArray<T> &predictions,
                                                     const nc::NdArray<T> &targets) const {
    return (-2.0 * (targets - predictions)) / static_cast<Scalar>(targets.numCols());
}

template<typename T>
T
perceptron::loss::SSR<T>::apply(const nc::NdArray<T> &predictions,
                                const nc::NdArray<T> &targets) const {
    size_t rows = predictions.numRows();
    size_t cols = predictions.numCols();

    auto loss = 0.0;
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            loss += std::pow(targets(i, j) - predictions(i, j), 2);

    return loss;
}


template<typename T>
nc::NdArray<T>
perceptron::loss::SSR<T>::differentiate(const nc::NdArray<T> &predictions,
                                        const nc::NdArray<T> &targets) const {
    return (-2.0 * (targets - predictions));
}

template<typename T>
T
perceptron::loss::CrossEntropy<T>::apply(const nc::NdArray<T> &predictions,
                                         const nc::NdArray<T> &targets) const {
    return nc::sum(nc::nan_to_num(-targets * nc::log(predictions) - (1.0 - targets) * nc::log(1.0 - predictions)))[0];
}

template<typename T>
nc::NdArray<T>
perceptron::loss::CrossEntropy<T>::differentiate(const nc::NdArray<T> &predictions,
                                                 const nc::NdArray<T> &targets) const {
    return predictions - targets;
}

template<typename T>
T
perceptron::loss::CrossEntropyLogits<T>::apply(const nc::NdArray<T> &predictions, const nc::NdArray<T> &targets) const {
    auto max_logits = nc::max(predictions, nc::Axis::ROW);
    auto exp_logits = nc::exp(predictions - max_logits);
    auto softmax = exp_logits / nc::sum(exp_logits, nc::Axis::ROW);
    auto clipped_softmax = nc::clip(softmax, 1e-15, 1.0 - 1e-15);
    auto log_likelihood = -nc::sum(targets * nc::log(clipped_softmax), nc::Axis::ROW);
    return nc::mean(log_likelihood)[0];
}

template<typename T>
nc::NdArray<T> perceptron::loss::CrossEntropyLogits<T>::differentiate(const nc::NdArray<T> &predictions,
                                                                      const nc::NdArray<T> &targets) const {
    auto max_logits = nc::max(predictions, nc::Axis::ROW);
    auto exp_logits = nc::exp(predictions - max_logits);
    auto softmax = exp_logits / nc::sum(exp_logits, nc::Axis::ROW);
    return softmax - targets;
}

template
class perceptron::loss::MeanSquaredError<Scalar>;

template
class perceptron::loss::SSR<Scalar>;

template
class perceptron::loss::CrossEntropy<Scalar>;

template
class perceptron::loss::CrossEntropyLogits<Scalar>;
