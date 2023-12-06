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
    size_t rows = predictions.numRows();
    size_t cols = predictions.numCols();

    throw std::runtime_error("unimplemented");
    return 0.0;
//    auto loss = 0.0;
//
//    for (int i = 0; i < trainData.Length; ++i)
//    {
//        Array.Copy(trainData[i], xValues, numInput); // get inputs
//        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get targets
//        double[] yValues = this.ComputeOutputs(xValues); // compute outputs
//        for (int j = 0; j < numOutput; ++j)
//        {
//            loss += Math.Log(yValues[j]) * tValues[j]; // CE error
//        }
//    }
//
//
//    return -1.0 * loss / trainData.Length;
}

template<typename T>
nc::NdArray<T>
perceptron::loss::CrossEntropy<T>::differentiate(const nc::NdArray<T> &predictions,
                                                 const nc::NdArray<T> &targets) const {
    return predictions - targets;
}

template
class perceptron::loss::MeanSquaredError<Scalar>;

template
class perceptron::loss::SSR<Scalar>;

template
class perceptron::loss::CrossEntropy<Scalar>;
