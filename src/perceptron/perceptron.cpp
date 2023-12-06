#include "perceptron/perceptron.hpp"

// This is required because we're not able to guarantee that the shape of the input data conforms
// to the shape of the weights. In numpy, this is done automatically, while using nc::NdArray,
// calling `nc::dot` performs usual matrix multiplication while using the `*` operator performs
// broadcast multiplication.
auto dot_or_broadcast_mult = [](const nc::NdArray<Scalar> &A, const nc::NdArray<Scalar> &B) {
    return A.numCols() == B.numRows() ? nc::dot(A, B) : A * B;
};

template<typename T>
auto min(const T a, const T b) { return a < b ? a : b; }

template<typename T>
auto max(const T a, const T b) { return a > b ? a : b; }

perceptron::Layer::Layer(const int input_size, const int output_size,
                         perceptron::activation::Activation<Scalar> &activation) :
        input_size(input_size),
        output_size(output_size),
        activation(activation) {}

perceptron::MultiLayerPerceptron::MultiLayerPerceptron(std::vector<Layer> layers,
                                                       perceptron::random::Random<Scalar> &randomizer)
        : layers(std::move(layers)) {
    for (auto &layer: this->layers) {
        auto weight = nc::NdArray<Scalar>(nc::Shape(layer.output_size, layer.input_size));
        for (auto &w: weight) w = randomizer.random();
        auto bias = nc::NdArray<Scalar>(nc::Shape(layer.output_size, 1));
        for (auto &w: bias) w = randomizer.random();
        weights.push_back(weight);
        biases.push_back(bias);
    }
}

perceptron::MultiLayerPerceptron::MultiLayerPerceptron(std::vector<Layer> layers,
                                                       std::vector<nc::NdArray<Scalar>> weights,
                                                       std::vector<nc::NdArray<Scalar>> biases)
        : layers(std::move(layers)),
          weights(std::move(weights)),
          biases(std::move(biases)) {}

void
perceptron::MultiLayerPerceptron::train(
        std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &train_data,
        const perceptron::loss::Loss<Scalar> &loss, const uint32_t max_epochs,
        const double learning_rate, const std::optional<OnEpochHandler> on_epoch_handler
) {
    for (auto epoch = 0; epoch < max_epochs; epoch++) {
        // Train the network by finding the gradient and update weights and biases on each epoch
        this->gradient_descent(train_data, loss, learning_rate);

        // Run the user-defined handler
        if (on_epoch_handler)
            if (on_epoch_handler.value()(epoch + 1, loss) == true)
                break;
    }
}

void perceptron::MultiLayerPerceptron::SGD(
        const std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &train_data,
        const perceptron::loss::Loss<Scalar> &loss, const uint32_t max_epochs,
        const double learning_rate, const uint32_t batch_size,
        const std::optional<OnEpochHandler> on_epoch_handler
) {
    // A random engine for shuffling the training data
    auto rng = std::default_random_engine(std::random_device()());

    for (auto epoch = 0; epoch < max_epochs; epoch++) {
        // Split the training data into a vector of batch_size
        auto batches = perceptron::split_vec_into_batch(train_data, batch_size);

        // Shuffle the training data to allow the network to learn
        // from a more representative data sample in each batch.
        std::shuffle(batches.begin(), batches.end(), rng);

        // Train the network with each batch (find the gradient and update weights and biases)
        for (auto &batch: batches)
            this->gradient_descent(batch, loss, learning_rate);

        // Run the user-defined handler
        if (on_epoch_handler)
            if (on_epoch_handler.value()(epoch + 1, loss) == true)
                break;
    }
}

void perceptron::MultiLayerPerceptron::gradient_descent(
        std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &batch,
        const perceptron::loss::Loss<Scalar> &loss,
        const double learning_rate) {
    // Create zero-ed weights and biases to keep track the delta and apply it later.
    auto nabla_w = std::vector<nc::NdArray<Scalar>>(weights.size());
    std::transform(weights.begin(), weights.end(), nabla_w.begin(),
                   [](const auto &weight) { return nc::zeros<Scalar>(weight.numRows(), weight.numCols()); });
    auto nabla_b = std::vector<nc::NdArray<Scalar>>(biases.size());
    std::transform(biases.begin(), biases.end(), nabla_b.begin(),
                   [](const auto &bias) { return nc::zeros<Scalar>(bias.numRows(), bias.numCols()); });

    // Backpropagate through the network for each training sample
    int i = -1; // an index to use for indexing within transform.
    for (auto &[train, target]: batch) {
        std::vector<nc::NdArray<Scalar>> delta_w, delta_b;
        std::tie(delta_w, delta_b) = this->backpropagate(train, target, loss);
        // Add the delta to each element
        std::transform(nabla_w.begin(), nabla_w.end(), nabla_w.begin(), [&](const auto &w) {
            i++;
            return w + delta_w[i];
        });
        i = -1;
        std::transform(nabla_b.begin(), nabla_b.end(), nabla_b.begin(), [&](const auto &b) {
            i++;
            return b + delta_b[i];
        });
        i = -1;
    }

    // Update weights and biases with respect to the learning rate and batch size
    std::transform(nabla_w.begin(), nabla_w.end(), weights.begin(), [&](const auto &nw) {
        i++;
        return weights[i] - (nw * (learning_rate / static_cast<double>(batch.size())));
    });
    i = -1;
    std::transform(nabla_b.begin(), nabla_b.end(), biases.begin(), [&](const auto &nb) {
        i++;
        return biases[i] - (nb * (learning_rate / static_cast<double>(batch.size())));
    });
}

std::pair<std::vector<nc::NdArray<Scalar>>, std::vector<nc::NdArray<Scalar>>>
perceptron::MultiLayerPerceptron::backpropagate(
        std::vector<Scalar> &train_data,
        std::vector<Scalar> &targets,
        const perceptron::loss::Loss<Scalar> &loss
) {
    // nabla stands for partial derivatives in calculus.
    std::vector<nc::NdArray<Scalar>> nabla_w(layers.size()), nabla_b(layers.size());

    // Forward pass
    auto [Z, A] = feedforward(train_data);
    A.insert(A.begin(), train_data);

    // Backward pass
    // Calculate the partial derivative of the cost with respect to activations.
    auto delta_loss = loss.differentiate(*(A.end() - 1), targets);
    // Calculate the partial derivative of the activation with respect to weighted sum (for the last layer only).
    auto delta_a_last = (layers.end() - 1)->activation.differentiate(*(Z.end() - 1));
    // Combine the partial derivatives (the last two terms).
    auto partial_delta = delta_a_last * delta_loss;

    // These are the partial derivatives for the last layer. Notice the index -1.
    *(nabla_w.end() - 1) = dot_or_broadcast_mult(partial_delta, (A.end() - 2)->transpose());
    *(nabla_b.end() - 1) = partial_delta;

    // After getting the derivative for the last part, we continue to find the derivatives for the entire network.
    for (int i = layers.size() - 2; i >= 0; i--) { // NOLINT(*-narrowing-conversions)
        // Notice the partial delta from the last layer got updated.
        partial_delta = dot_or_broadcast_mult(weights[i + 1].transpose(), partial_delta) *
                        layers[i].activation.differentiate(Z[i]);
        auto activation = A[max(0, i - 1)].transpose();
        nabla_w[i] = dot_or_broadcast_mult(partial_delta, activation);
        nabla_b[i] = partial_delta;
    }

    return std::make_pair(nabla_w, nabla_b);
}

std::pair<std::vector<nc::NdArray<Scalar>>, std::vector<nc::NdArray<Scalar>>>
perceptron::MultiLayerPerceptron::feedforward(const nc::NdArray<Scalar> &input) {
    std::vector<nc::NdArray<Scalar>> Z(layers.size()), A(layers.size());
    // make the first activation as the train data because it is used as the input to the first layer,
    // it will be later override by the actual activation.
    A[0] = input.transpose();
    for (int i = 0; i < layers.size(); i++) {
        Z[i] = dot_or_broadcast_mult(weights[i], A[max(0, i - 1)]) + biases[i];
        A[i] = layers[i].activation.apply(Z[i]);
    }

    return std::make_pair(Z, A);
}

nc::NdArray<Scalar>
perceptron::MultiLayerPerceptron::predict(const nc::NdArray<Scalar> &input) {
    return feedforward(input).second.back();
}

perceptron::Evaluation
perceptron::MultiLayerPerceptron::evaluate(
        std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> &_inputs,
        const perceptron::loss::Loss<Scalar> &loss) {
    auto predictions = nc::NdArray<Scalar>();
    auto targets = nc::NdArray<Scalar>();
    for (auto &[input, target]: _inputs) {
        predictions = nc::append(predictions, this->predict(input));
        targets = nc::append(targets, nc::NdArray(target));
    }

    Evaluation evaluation{
            .predictions = predictions,
            .targets = targets,
            .error = loss.apply(predictions, targets),
            .accuracy = nc::count_nonzero(nc::equal(predictions, targets))[0] / static_cast<double> (targets.size())
    };

    return evaluation;
}
