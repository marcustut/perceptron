#include <gtest/gtest.h>

#include "perceptron/loss.hpp"

#define ABS_ERROR 0.001

struct LossTC {
    std::vector<std::vector<Scalar>> targets, predictions;
};

TEST(loss, MeanSquaredError) {
    auto loss = perceptron::loss::MeanSquaredError<double>();

    auto tcs = std::vector<std::pair<LossTC, std::pair<double, std::vector<std::vector<double>>>>>{
            std::make_pair(
                    LossTC{
                            .targets = std::vector<std::vector<Scalar>>{
                                    {0},
                                    {1},
                                    {1},
                                    {0}
                            },
                            .predictions = std::vector<std::vector<Scalar>>{
                                    {0.012312},
                                    {1.1230},
                                    {1.012031},
                                    {0.12346198}
                            }
                    }, std::make_pair(0.0076670, std::vector<std::vector<double>>{
                            {0.024624},
                            {0.246},
                            {0.024061},
                            {0.24692}})),
    };
    for (auto &tc: tcs) {
        ASSERT_NEAR(tc.second.first, loss.apply(nc::NdArray(tc.first.predictions), nc::NdArray(tc.first.targets)),
                    ABS_ERROR);

        auto gradients = loss.differentiate(nc::NdArray(tc.first.predictions), nc::NdArray(tc.first.targets));
        for (size_t i = 0; i < gradients.numRows(); i++)
            for (size_t j = 0; j < gradients.numCols(); j++)
                ASSERT_NEAR(tc.second.second[i][j], gradients(i, j), ABS_ERROR);
    }
}

TEST(loss, SSR) {
    auto loss = perceptron::loss::SSR<double>();

    auto tcs = std::vector<std::pair<LossTC, std::pair<double, std::vector<std::vector<double>>>>>{
            std::make_pair(
                    LossTC{
                            .targets = std::vector<std::vector<Scalar>>{
                                    {0},
                                    {1},
                                    {1},
                                    {0}
                            },
                            .predictions = std::vector<std::vector<Scalar>>{
                                    {0.012312},
                                    {1.1230},
                                    {1.012031},
                                    {0.12346198}
                            }
                    }, std::make_pair(0.030668, std::vector<std::vector<double>>{
                            {0.024624},
                            {0.246},
                            {0.024061},
                            {0.24692}})),
    };
    for (auto &tc: tcs) {
        ASSERT_NEAR(tc.second.first, loss.apply(nc::NdArray(tc.first.predictions), nc::NdArray(tc.first.targets)),
                    ABS_ERROR);

        auto gradients = loss.differentiate(nc::NdArray(tc.first.predictions), nc::NdArray(tc.first.targets));
        for (size_t i = 0; i < gradients.numRows(); i++)
            for (size_t j = 0; j < gradients.numCols(); j++)
                ASSERT_NEAR(tc.second.second[i][j], gradients(i, j), ABS_ERROR);
    }
}
