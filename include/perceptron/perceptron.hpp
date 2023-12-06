#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>

#include "perceptron/random.hpp"
#include "perceptron/activation.hpp"
#include "perceptron/loss.hpp"
#include "perceptron/blas.hpp"

#define FMT_HEADER_ONLY

#include "fmt/format.h"
#include "fmt/ranges.h"

#include "NumCpp.hpp"

using Matrix = perceptron::blas::Matrix;
using OnEpochHandler = std::function<std::optional<bool>(const int, const perceptron::loss::Loss<Scalar> &)>;

namespace perceptron {
    struct Evaluation {
        nc::NdArray<Scalar> predictions;
        nc::NdArray<Scalar> targets;
        Scalar error, accuracy;
    };  // struct Evaluation

    /**
     * Layer is just a collection of Perceptron that can be used as input / hidden / output layers.
     */
    class Layer {
    public:
        // Number of input neurons
        const int input_size;
        // Number of output neurons
        const int output_size;
        // The activation function used in this layer.
        perceptron::activation::Activation<Scalar> &activation;

        Layer(int input_size, int output_size, perceptron::activation::Activation<Scalar> &activation);
    };

    /**
     * MultiLayerPerceptron is also known as "Neural Networks" which consists of multiple layers of
     * perceptron or so called "Neurons".
     */
    class MultiLayerPerceptron {
    public:
        MultiLayerPerceptron(std::vector<Layer> layers, perceptron::random::Random<Scalar> &randomizer);

        MultiLayerPerceptron(std::vector<Layer> layers, std::vector<nc::NdArray<Scalar>> weights,
                             std::vector<nc::NdArray<Scalar>> biases);

        void train(std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &train_data,
                   const perceptron::loss::Loss<Scalar> &loss, const uint32_t max_epochs, const double learning_rate,
                   const std::optional<OnEpochHandler> on_epoch_handler = std::nullopt);

        void SGD(const std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &train_data,
                 const perceptron::loss::Loss<Scalar> &loss, const uint32_t max_epochs, const double learning_rate,
                 const uint32_t batch_size, const std::optional<OnEpochHandler> on_epoch_handler = std::nullopt);

        nc::NdArray<Scalar> predict(const nc::NdArray<Scalar> &input);

        Evaluation evaluate(std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &inputs,
                            const perceptron::loss::Loss<Scalar> &loss);

    private:
        std::vector<Layer> layers;
        std::vector<nc::NdArray<Scalar>> weights; // Weights for each layer
        std::vector<nc::NdArray<Scalar>> biases; // Biases for each layer

        std::pair<std::vector<nc::NdArray<Scalar>>, std::vector<nc::NdArray<Scalar>>>
        feedforward(const nc::NdArray<Scalar> &input);

        void gradient_descent(std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &batch,
                              const perceptron::loss::Loss<Scalar> &loss,
                              const double learning_rate);

        std::pair<std::vector<nc::NdArray<Scalar>>, std::vector<nc::NdArray<Scalar>>> backpropagate(
                std::vector<Scalar> &train_data,
                std::vector<Scalar> &targets,
                const perceptron::loss::Loss<Scalar> &loss
        );

    };

    template<typename T>
    auto split_vec_into_batch(const std::vector<T> &vec, const uint32_t batch_size) {
        std::vector<std::vector<T>> batches(ceil((double) vec.size() / (double) batch_size));

        int idx = 0;
        for (auto i = 0; i < vec.size(); i++) {
            batches[idx].push_back(vec[i]);
            if (i % batch_size == batch_size - 1) idx++;
        }

        return batches;
    };
} // namespace perceptron

#endif //PERCEPTRON_H
