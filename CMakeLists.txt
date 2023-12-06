cmake_minimum_required(VERSION 3.26)

project(perceptron
        LANGUAGES CXX
        DESCRIPTION "Multi-layer perceptron in C++"
)
set(CMAKE_CXX_STANDARD 20)

option(PERCEPTRON_BUILD_EXAMPLES "Set to ON to build examples" ON)
option(PERCEPTRON_BUILD_TESTS "Set to ON to build tests" OFF)

add_library(perceptron
        include/perceptron/perceptron.hpp
        include/perceptron/activation.hpp
        include/perceptron/loss.hpp
        include/perceptron/random.hpp
        include/perceptron/blas.hpp
        src/perceptron/perceptron.cpp
        src/perceptron/activation.cpp
        src/perceptron/loss.cpp
        src/perceptron/random.cpp
        src/perceptron/blas.cpp
        include/NumCpp.hpp
)
target_include_directories(perceptron PRIVATE ${PROJECT_SOURCE_DIR}/include)

# use libfmt
find_package(fmt)
target_link_libraries(perceptron fmt::fmt)

add_subdirectory(tests)
add_subdirectory(examples)