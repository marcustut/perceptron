#include <gtest/gtest.h>

#include "perceptron/activation.hpp"

#define ABS_ERROR 0.001

TEST(activation, linear) {
    auto activation = perceptron::activation::Linear<double>();

    auto tcs = std::vector<std::pair<double, std::pair<double, double>>>{
            std::make_pair(1, std::make_pair(1, 1)),
            std::make_pair(2, std::make_pair(2, 1)),
    };
    for (auto &tc: tcs) {
        ASSERT_NEAR(tc.second.first, activation.apply(tc.first), ABS_ERROR);
        ASSERT_NEAR(tc.second.second, activation.differentiate(tc.first), ABS_ERROR);
    }
}

TEST(activation, sigmoid) {
    auto activation = perceptron::activation::Sigmoid<double>();

    auto tcs = std::vector<std::pair<double, std::pair<double, double>>>{
            std::make_pair(1, std::make_pair(0.73105, 0.19661)),
            std::make_pair(2, std::make_pair(0.88079, 0.10499)),
    };
    for (auto &tc: tcs) {
        ASSERT_NEAR(tc.second.first, activation.apply(tc.first), ABS_ERROR);
        ASSERT_NEAR(tc.second.second, activation.differentiate(tc.first), ABS_ERROR);
    }
}

TEST(activation, relu) {
    auto activation = perceptron::activation::ReLU<double>();

    auto tcs = std::vector<std::pair<double, std::pair<double, double>>>{
            std::make_pair(-1, std::make_pair(0, 0)),
            std::make_pair(0, std::make_pair(0, 0)),
            std::make_pair(1, std::make_pair(1, 1)),
            std::make_pair(2, std::make_pair(2, 1)),
            std::make_pair(3, std::make_pair(3, 1)),
    };
    for (auto &tc: tcs) {
        ASSERT_NEAR(tc.second.first, activation.apply(tc.first), ABS_ERROR);
        ASSERT_NEAR(tc.second.second, activation.differentiate(tc.first), ABS_ERROR);
    }
}

TEST(activation, tanh) {
    auto activation = perceptron::activation::TanH<double>();

    auto tcs = std::vector<std::pair<double, std::pair<double, double>>>{
            std::make_pair(-1, std::make_pair(-0.76159, 0.41997)),
            std::make_pair(0, std::make_pair(0, 1)),
            std::make_pair(1, std::make_pair(0.76159, 0.41997)),
            std::make_pair(2, std::make_pair(0.96402, 0.07065)),
            std::make_pair(3, std::make_pair(0.99505, 0.00986)),
    };
    for (auto &tc: tcs) {
        ASSERT_NEAR(tc.second.first, activation.apply(tc.first), ABS_ERROR);
        ASSERT_NEAR(tc.second.second, activation.differentiate(tc.first), ABS_ERROR);
    }
}
