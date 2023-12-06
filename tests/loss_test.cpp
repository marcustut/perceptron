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

TEST(loss, CrossEntropyLogits) {
    auto loss = perceptron::loss::CrossEntropyLogits<double>();

    auto tcs = std::vector<std::pair<LossTC, std::pair<double, std::vector<std::vector<double>>>>>{
            std::make_pair(
                    LossTC{
                            .targets = std::vector<std::vector<Scalar>>{
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {1},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0},
                                    {0}
                            },
                            .predictions = std::vector<std::vector<Scalar>>{
                                    {0.16392685927613965},
                                    {-0.5706169985274203},
                                    {0.8422551527947455},
                                    {0.46304997884536786},
                                    {-0.03422489964111705},
                                    {0.38704445542196425},
                                    {-1.2760711150878015},
                                    {0.8815271290466552},
                                    {0.7659245198842937},
                                    {0.26438481281522647},
                                    {-0.8989946096774751},
                                    {-0.13660986885870188},
                                    {0.8783139880496933},
                                    {-0.43523791629259906},
                                    {0.7587406465491344},
                                    {-0.8377580054197031},
                                    {0.7032827598796619},
                                    {-0.1309133183139091},
                                    {0.5236785926370521},
                                    {0.6888396391569431},
                                    {0.6212406686065461},
                                    {0.3372102246213558},
                                    {-0.03499552594580502},
                                    {1.045891819962511},
                                    {0.5301360884556183},
                                    {0.5647164343444457}
                            }
                    },
                    std::make_pair(4.920236, std::vector<std::vector<double>>{
                            {0.0308001},
                            {0.0147755},
                            {0.060694},
                            {0.0415393},
                            {0.0252636},
                            {0.0384991},
                            {-0.992703},
                            {0.063125},
                            {0.0562336},
                            {0.0340549},
                            {0.0106397},
                            {0.022805},
                            {0.0629225},
                            {0.0169176},
                            {0.0558311},
                            {0.0113116},
                            {0.0528191},
                            {0.0229353},
                            {0.0441357},
                            {0.0520617},
                            {0.0486587},
                            {0.0366275},
                            {0.0252441},
                            {0.0744019},
                            {0.0444216},
                            {0.0459846}
                    })),
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