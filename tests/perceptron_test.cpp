#include <gtest/gtest.h>

#include "perceptron/perceptron.hpp"
#include "fmt/ranges.h"

#include "fmtlog/fmtlog.h"
#include "fmtlog/fmtlog-inl.h"

#define ABS_ERROR 0.05

struct LayerTC {
    int input_size;
    int output_size;
    perceptron::activation::Activation<double> *activation;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> inputs;
};

//TEST(perceptron, layer_forward) {
//    auto sigmoid = perceptron::activation::Sigmoid<double>();
//    auto linear = perceptron::activation::Linear<double>();
//    auto tcs = std::vector<std::pair<LayerTC, std::vector<double>>>{
//            std::make_pair(
//                    LayerTC{.input_size = 1,
//                            .output_size = 2,
//                            .activation = &linear,
//                            .weights = std::vector<std::vector<double>>{{-34.4, -2.52}},
//                            .biases = std::vector<double>{{2.14, 1.29}},
//                            .inputs = std::vector<double>{0.5},
//                    },
//                    std::vector<double>{-15.06, 0.03}
//            ),
//            std::make_pair(
//                    LayerTC{.input_size = 1,
//                            .output_size = 2,
//                            .activation = &sigmoid,
//                            .weights = std::vector<std::vector<double>>{{-34.4, -2.52}},
//                            .biases = std::vector<double>{{2.14, 1.29}},
//                            .inputs = std::vector<double>{0.5},
//                    },
//                    std::vector<double>{2.88e-7, 0.50749}
//            ),
//    };
//
//    for (auto &tc: tcs) {
//        auto results = perceptron::Layer(
//                tc.first.input_size,
//                tc.first.output_size,
//                *tc.first.activation,
//                tc.first.weights,
//                tc.first.biases
//        ).forward(tc.first.inputs);
//        for (size_t i = 0; i < results.second.size(); i++)
//            ASSERT_NEAR(tc.second[i], results.second[i], ABS_ERROR);
//    }
//}

//TEST(perceptron, feed_forward_network) {
//    auto sigmoid = perceptron::activation::Sigmoid<double>();
//    std::vector<std::vector<double>> TRAIN_DATA = {
//            {0, 0},
//            {0, 1},
//            {1, 0},
//            {1, 1}
//    };
//    auto randomizer = perceptron::random::Choice(std::vector<double>{1.0});
//    auto layers = std::vector<perceptron::Layer>{
//            perceptron::Layer(2, 2, sigmoid),
//            perceptron::Layer(2, 1, sigmoid)};
//    auto mlp = *new perceptron::MultiLayerPerceptron(layers, randomizer);
//    ASSERT_NEAR(0.7310585786, mlp.predict(TRAIN_DATA[0])[0], ABS_ERROR);
//    ASSERT_NEAR(0.8118562749, mlp.predict(TRAIN_DATA[1])[0], ABS_ERROR);
//    ASSERT_NEAR(0.8118562749, mlp.predict(TRAIN_DATA[2])[0], ABS_ERROR);
//    ASSERT_NEAR(0.8534092046, mlp.predict(TRAIN_DATA[3])[0], ABS_ERROR);
//}
