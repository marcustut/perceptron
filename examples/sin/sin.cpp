
#include <utility>
#include <fmt/ranges.h>

#include "perceptron/perceptron.hpp"

#define IN_FEATURES 4
#define OUT_FEATURES 1
#define HIDDEN_FEATURES 8

#define MAX_EPOCHS 10000
#define LEARNING_RATE 0.001
#define TARGET_ERROR 0.001
#define BATCH_SIZE 4

#define EXPORT_TARGETS_PATH "sin_targets.csv"
#define EXPORT_PREDICTIONS_PATH "sin_predictions.csv"

auto generate_data(const int n) {
    std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> train_data;
    auto randomizer = perceptron::random::Uniform<Scalar>(-1, 1);
    for (int i = 0; i < n; i++) {
        auto x1 = randomizer.random();
        auto x2 = randomizer.random();
        auto x3 = randomizer.random();
        auto x4 = randomizer.random();
        train_data.emplace_back(
                std::vector<Scalar>{x1, x2, x3, x4},
                std::vector<Scalar>{std::sin(x1 - x2 + x3 - x4)}
        );
    }
    return train_data;
}

int main() {
    auto train_data = generate_data(400);
    auto test_data = generate_data(100);

    auto randomizer = perceptron::random::Xavier<Scalar>(IN_FEATURES, OUT_FEATURES);
    auto activation = perceptron::activation::TanH<Scalar>();
    auto mlp = perceptron::MultiLayerPerceptron(
            std::vector<perceptron::Layer>{
                    perceptron::Layer(IN_FEATURES, HIDDEN_FEATURES, activation),
                    perceptron::Layer(HIDDEN_FEATURES, OUT_FEATURES, activation)
            },
            randomizer
    );

    auto loss = perceptron::loss::SSR<Scalar>();
    auto on_epoch_handler = [&](const int epoch, const perceptron::loss::Loss<Scalar> &loss) {
        auto evaluation = mlp.evaluate(train_data, loss);
        if (evaluation.error < TARGET_ERROR) {
            fmt::println("Error is less than target error ({}). Stopping...", TARGET_ERROR);
            return true;
        }
        if (epoch % 100 == 0)
            fmt::println("Epoch {}: error is {}", epoch, evaluation.error);
        return false;
    };

//    mlp.train(train_data, loss, MAX_EPOCHS, LEARNING_RATE, on_epoch_handler);
    mlp.SGD(train_data, loss, MAX_EPOCHS, LEARNING_RATE, BATCH_SIZE, on_epoch_handler);

    auto targets = nc::NdArray<Scalar>();
    auto predictions = nc::NdArray<Scalar>();

    for (auto &[input, target]: test_data) {
        auto prediction = mlp.predict(input);
        fmt::println("Input: {}: Expected {}, Got {}", input, target, prediction);
        targets = nc::append(targets, nc::NdArray(target));
        predictions = nc::append(predictions, prediction);
    }

    nc::tofile(targets, EXPORT_TARGETS_PATH, ',');
    nc::tofile(predictions, EXPORT_PREDICTIONS_PATH, ',');

    return 0;
}
