#ifndef PERCEPTRON_RANDOM_HPP
#define PERCEPTRON_RANDOM_HPP

#include <random>

namespace perceptron::random {
    /**
     * An interface for all weight initializer
     * @tparam T - type of the number to generate eg. `double`, `int`, etc.
     */
    template<typename T>
    class Random {
    public:
        virtual T random() = 0;
    };

    template<typename T>
    class Uniform : public Random<T> {
    public:
        Uniform(T lower_bound, T upper_bound);

        T random();

    private:
        std::uniform_real_distribution<T> distribution;
        std::default_random_engine engine;
    };

    template<typename T>
    class Gaussian : public Random<T> {
    public:
        Gaussian(T mean, T variance);

        T random();

    private:
        std::normal_distribution<T> distribution;
        std::default_random_engine engine;
    };

    template<typename T>
    class Xavier : public Random<T> {
    public:
        Xavier(T input_size, T output_size);

        T random();

    private:
        std::normal_distribution<T> distribution;
        std::default_random_engine engine;
    };

    template<typename T>
    class Choice : public Random<T> {
    public:
        Choice(std::vector<T> choices);

        T random();

        std::vector<T> random_vector(size_t n);

        std::vector<std::vector<T>> random_matrix(size_t m, size_t n);

    private:
        std::vector<T> choices;
        std::mt19937 generator;
        std::uniform_int_distribution<> distribution;
    };

//template<typename T>
//class XavierWeight : Weight<T> {
//public:
//    explicit XavierWeight(int n);
//
//    std::vector<T> generate();
//
//private:
//    std::uniform_real_distribution<T> distribution;
//    std::default_random_engine engine;
//};

} // namespace perceptron::random

#endif //PERCEPTRON_RANDOM_HPP

