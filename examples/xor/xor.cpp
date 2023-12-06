
#include <utility>
#include <fmt/ranges.h>

#include "perceptron/perceptron.hpp"

#define IN_FEATURES 2
#define OUT_FEATURES 1
#define HIDDEN_FEATURES 4

#define MAX_EPOCHS 10000
#define LEARNING_RATE 3.0
#define TARGET_ERROR 0.001
#define BATCH_SIZE 2

std::vector<std::pair<std::vector<Scalar>, std::vector<Scalar>>> TRAIN_DATA = {
        std::make_pair(std::vector<Scalar>{0, 0}, std::vector<Scalar>{0}),
        std::make_pair(std::vector<Scalar>{0, 1}, std::vector<Scalar>{1}),
        std::make_pair(std::vector<Scalar>{1, 0}, std::vector<Scalar>{1}),
        std::make_pair(std::vector<Scalar>{1, 1}, std::vector<Scalar>{0}),
};

int main() {
    auto randomizer = perceptron::random::Xavier<Scalar>(IN_FEATURES, OUT_FEATURES);
    auto activation = perceptron::activation::Sigmoid<Scalar>();
    auto mlp = perceptron::MultiLayerPerceptron(
            std::vector<perceptron::Layer>{
                    perceptron::Layer(IN_FEATURES, HIDDEN_FEATURES, activation),
                    perceptron::Layer(HIDDEN_FEATURES, OUT_FEATURES, activation)
            },
            randomizer
    );

    auto loss = perceptron::loss::MeanSquaredError<Scalar>();
    auto on_epoch_handler = [&](const int epoch, const perceptron::loss::Loss<Scalar> &loss) {
        auto evaluation = mlp.evaluate(TRAIN_DATA, loss);
        if (evaluation.error < TARGET_ERROR) {
            fmt::println("Error is less than target error ({}). Stopping...", TARGET_ERROR);
            return true;
        }
        fmt::println("Epoch {}: {} (error)", epoch, evaluation.error);
        return false;
    };

//    mlp.train(TRAIN_DATA, loss, MAX_EPOCHS, LEARNING_RATE, on_epoch_handler);
    mlp.SGD(TRAIN_DATA, loss, MAX_EPOCHS, LEARNING_RATE, BATCH_SIZE, on_epoch_handler);

    for (auto &[data, _]: TRAIN_DATA)
        fmt::println("{}: {}", data, mlp.predict(data));

    return 0;
}
