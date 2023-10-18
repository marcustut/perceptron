cmake_minimum_required(VERSION 3.26)

project(perceptron
        LANGUAGES CXX
        DESCRIPTION "Multi-layer perceptron in C++"
)

option(PERCEPTRON_BUILD_EXAMPLES "Set to ON to build examples" OFF)
option(PERCEPTRON_BUILD_TESTS "Set to ON to build tests" OFF)

add_library(perceptron
        include/perceptron/perceptron.hpp
        src/perceptron/perceptron.cpp
)
target_include_directories(perceptron PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(tests)
add_subdirectory(examples)