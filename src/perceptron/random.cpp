#include "perceptron/random.hpp"
#include "perceptron/blas.hpp"

#include <chrono>

template<typename T>
perceptron::random::Uniform<T>::Uniform(T lower_bound, T upper_bound) :
        distribution(std::uniform_real_distribution<T>(lower_bound, upper_bound)),
        // Construct a random engine using current timestamp as seed.
        engine(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count())) {}

template<typename T>
T perceptron::random::Uniform<T>::random() {
    return this->distribution(this->engine);
}

template<typename T>
perceptron::random::Gaussian<T>::Gaussian(T mean, T variance) :
        distribution(std::normal_distribution<T>(mean, variance)),
        // Construct a random engine using current timestamp as seed.
        engine(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count())) {}

template<typename T>
T perceptron::random::Gaussian<T>::random() {
    return this->distribution(this->engine);
}

template<typename T>
perceptron::random::Xavier<T>::Xavier(T input_size, T output_size) :
        distribution(std::normal_distribution<T>(0.0, std::sqrt(2.0 / (input_size + output_size)))),
        // Construct a random engine using current timestamp as seed.
        engine(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count())) {}

template<typename T>
T perceptron::random::Xavier<T>::random() {
    return this->distribution(this->engine);
}

template<typename T>
perceptron::random::Choice<T>::Choice(std::vector<T> choices) :
        choices(choices),
        generator(std::mt19937(std::random_device()())),
        distribution(std::uniform_int_distribution<>(0, choices.size() - 1)) {}

template<typename T>
T perceptron::random::Choice<T>::random() {
    return this->choices[this->distribution(this->generator)];
}

template
class perceptron::random::Uniform<Scalar>;

template
class perceptron::random::Gaussian<Scalar>;

template
class perceptron::random::Xavier<Scalar>;

template
class perceptron::random::Choice<Scalar>;
