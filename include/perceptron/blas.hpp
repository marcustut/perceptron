#ifndef PERCEPTRON_BLAS_H
#define PERCEPTRON_BLAS_H

#include <vector>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <fmt/ranges.h>

#include "perceptron/random.hpp"

typedef double Scalar;

namespace perceptron::blas {
    class Blas {
    public:
        static std::vector<Scalar> add(const std::vector<Scalar> &x, const Scalar y) {
            auto out = x;
            for (auto &it: out) it += y;
            return out;
        }

        static std::vector<Scalar> add(const std::vector<Scalar> &x, const std::vector<Scalar> &y) {
            assert(x.size() == y.size());
            auto out = x;
            for (size_t i = 0; i < x.size(); i++) out[i] += y[i];
            return out;
        }

        static std::vector<Scalar> subtract(const std::vector<Scalar> &x, const Scalar y) {
            auto out = x;
            for (auto &it: out) it -= y;
            return out;
        }

        static std::vector<Scalar> subtract(const std::vector<Scalar> &x, const std::vector<Scalar> &y) {
            assert(x.size() == y.size());
            auto out = x;
            for (size_t i = 0; i < x.size(); i++) out[i] -= y[i];
            return out;
        }

        static std::vector<Scalar> dot(const std::vector<Scalar> &x, const Scalar y) {
            auto out = x;
            for (auto &it: out) it *= y;
            return out;
        }

        static std::vector<Scalar> dot(const std::vector<Scalar> &x, const std::vector<Scalar> &y) {
            assert(x.size() == y.size());
            auto out = x;
            for (size_t i = 0; i < x.size(); i++) out[i] *= y[i];
            return out;
        }

        static std::vector<Scalar> divide(const std::vector<Scalar> &x, const Scalar y) {
            auto out = x;
            for (auto &it: out) it /= y;
            return out;
        }

        static std::vector<Scalar> divide(const std::vector<Scalar> &x, const std::vector<Scalar> &y) {
            assert(x.size() == y.size());
            auto out = x;
            for (size_t i = 0; i < x.size(); i++) out[i] /= y[i];
            return out;
        }

        static Scalar nan_to_num(Scalar num) {
            if (isinf(num))
                if (signbit(num))
                    return -std::numeric_limits<Scalar>::max();
                else
                    return std::numeric_limits<Scalar>::max();
            else
                return num;
        }
    };

    class Matrix {
    public:
        // Create a rows-by-cols matrix from 2D vectors.
        static Matrix from(std::vector<std::vector<Scalar>> matrix) {
            assert(!matrix.empty());
            Matrix m(matrix.size(), matrix[0].size());

            for (std::size_t row = 0; row < m.rows(); ++row)
                for (std::size_t col = 0; col < m.cols(); ++col)
                    m(row, col) = matrix[row][col];

            return m;
        }

        // Create a rows-by-cols matrix filled with random numbers using a randomizer.
        static Matrix random(size_t rows, size_t cols, perceptron::random::Random<Scalar> &randomizer);

        // Create a rows-by-cols matrix filled with zeroes.
        static Matrix zeroes(size_t rows, size_t cols);

        // Create a rows-by-cols matrix filled with ones.
        static Matrix ones(size_t rows, size_t cols);

        // dot is the actual normal dot product which performs matrix multiplication,
        // but the operator* performs broadcasting.
        static Matrix dot(const Matrix &A, const Matrix &B);

        // Build a default 0x0 matrix.
        Matrix();

        // Build an uninitialized rows-by-cols matrix.
        Matrix(size_t rows, size_t cols);

        // Build an initialized rows-by-cols matrix.
        Matrix(size_t rows, size_t cols, Scalar default_val);

        // Build matrix with an initializer list.
        Matrix(std::initializer_list<std::initializer_list<Scalar>> lists);

        // Return number of rows
        [[nodiscard]] size_t rows() const { return m_rows; }

        // Return number of columns
        [[nodiscard]] size_t cols() const { return m_cols; }

        auto begin() { return m_data.begin(); }

        auto end() { return m_data.end(); }

        [[nodiscard]] std::vector<std::vector<Scalar>> to_vector() const {
            auto out = std::vector<std::vector<Scalar>>(rows(), std::vector<Scalar>(cols()));
            for (size_t i = 0; i < rows(); i++)
                for (size_t j = 0; j < cols(); j++)
                    out[i][j] = (*this)(i, j);
            return out;
        }

        Matrix transpose() const {
            Matrix out(m_cols, m_rows);
            for (size_t i = 0; i < rows(); i++)
                for (size_t j = 0; j < cols(); j++)
                    out(j, i) = (*this)(i, j);
            return out;
        }

        // Value at (row, col)
        Scalar operator()(size_t row, size_t col) const {
            assert(row < rows());
            assert(col < cols());
            return m_data[row * cols() + col];
        }

        // Reference to value at (row, col)
        Scalar &operator()(size_t row, size_t col) {
            assert(row < rows());
            assert(col < cols());
            return m_data[row * cols() + col];
        }

        // Matrix x Matrix
        // This does broadcast multiplication, if a dot product is desired use `Matrix::dot`.
        Matrix operator*(const Matrix &other) const;

        Matrix operator*(const std::vector<std::vector<Scalar>> &other) const;

        // Matrix + Matrix
        Matrix operator+(const Matrix &other) const;

        // Matrix - Matrix
        Matrix operator-(const Matrix &other) const;

        // Matrix x Vector
        std::vector<Scalar> operator*(const std::vector<Scalar> &other) const;

        // Matrix + Vector
        Matrix operator+(const std::vector<Scalar> &other) const;

        // Matrix x Scalar
        Matrix operator*(const Scalar &other) const;

        // Matrix + Scalar
        Matrix operator+(const Scalar &other) const;

        // Matrix - Scalar
        Matrix operator-(const Scalar &other) const;

        // Matrix / Scalar
        Matrix operator/(const Scalar &other) const;

    private:
        Matrix(size_t rows, size_t cols, const std::vector<Scalar> &old);

        std::vector<Scalar> m_data;
        std::size_t m_rows;
        std::size_t m_cols;
    };

    std::ostream &operator<<(std::ostream &os, const Matrix &m);

} // namespace perceptron::blas

template<>
struct fmt::formatter<perceptron::blas::Matrix> : formatter<std::vector<std::vector<Scalar>>> {
    auto format(const perceptron::blas::Matrix &n, format_context &ctx) const -> format_context::iterator {
        return formatter<std::vector<std::vector<Scalar>>>::format(n.to_vector(), ctx);
    }
};


#endif //PERCEPTRON_BLAS_H
