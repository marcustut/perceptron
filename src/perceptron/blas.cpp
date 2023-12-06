#include "perceptron/blas.hpp"

namespace perceptron::blas {
    Matrix::Matrix() : m_data(std::vector<Scalar>()), m_rows(0), m_cols(0) {}

    Matrix::Matrix(size_t rows, size_t cols) : m_data(std::vector<Scalar>(rows * cols)),
                                               m_rows(rows),
                                               m_cols(cols) {}

    Matrix::Matrix(size_t rows, size_t cols, Scalar default_val)
            : m_data(std::vector<Scalar>(rows * cols, default_val)),
              m_rows(rows),
              m_cols(cols) {}

    Matrix::Matrix(std::initializer_list<std::initializer_list<Scalar>> lists) :
            Matrix(lists.size(), lists.size() ? lists.begin()->size() : 0) {
        int i = 0, j = 0;
        for (const auto &list: lists) {
            for (const auto &v: list) {
                (*this)(i, j) = v;
                ++j;
            }
            j = 0;
            ++i;
        }
    }

    Matrix::Matrix(size_t rows, size_t cols, const std::vector<Scalar> &old) : m_data{std::vector<Scalar>(old)},
                                                                               m_rows{rows},
                                                                               m_cols{cols} {}

    Matrix Matrix::random(size_t rows, size_t cols,
                          perceptron::random::Random<Scalar> &randomizer) {
        Matrix m(rows, cols);

        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t col = 0; col < cols; ++col)
                m(row, col) = randomizer.random();

        return m;
    }

    Matrix Matrix::zeroes(size_t rows, size_t cols) {
        return Matrix::from(std::vector<std::vector<Scalar>>(rows, std::vector<Scalar>(cols, 0.0)));
    }

    Matrix Matrix::ones(size_t rows, size_t cols) {
        return Matrix::from(std::vector<std::vector<Scalar>>(rows, std::vector<Scalar>(cols, 1.0)));
    }

    Matrix Matrix::dot(const Matrix &A, const Matrix &B) {
        assert(A.cols() == B.rows());

        Matrix out(A.rows(), B.cols());

        for (std::size_t i = 0; i < A.rows(); ++i)
            for (std::size_t j = 0; j < B.cols(); ++j) {
                Scalar sum = 0;
                for (std::size_t k = 0; k < A.cols(); ++k)
                    sum += A(i, k) * B(k, j);
                out(i, j) = sum;
            }

        return out;
    }

    Matrix Matrix::operator*(const Matrix &other) const {
        Matrix out = Matrix(
                static_cast<size_t>(fmax(rows(), other.rows())),
                static_cast<size_t>(fmax(cols(), other.cols())));

        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j) {
                Scalar a = (i < rows() && j < cols()) ? (*this)(i, j) : 1.0;
                Scalar b = (i < other.rows() && j < other.cols()) ? other(i, j) : 1.0;
                out(i, j) = a * b;
            }

        return out;
    }

    Matrix Matrix::operator*(const std::vector<std::vector<Scalar>> &other) const {
        return (*this) * Matrix::from(other);
    }

    std::vector<Scalar> Matrix::operator*(const std::vector<Scalar> &other) const {
        assert(cols() == other.size());

        std::vector<Scalar> out(rows(), 0.0);

        for (size_t i = 0; i < rows(); i++)
            for (size_t j = 0; j < cols(); j++)
                out[i] += other[j] * (*this)(i, j);

        return out;
    }

    Matrix Matrix::operator+(const std::vector<Scalar> &other) const {
        auto out = Matrix(rows(), static_cast<size_t>(fmax(cols(), other.size())));

        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j) {
                Scalar a = (j < cols()) ? (*this)(i, j) : 0.0;
                Scalar b = (j < other.size()) ? other[j] : 0.0;
                out(i, j) = a + b;
            }

        return out;
    }


    Matrix Matrix::operator+(const Matrix &other) const {
        Matrix out(
                static_cast<size_t>(fmax(rows(), other.rows())),
                static_cast<size_t>(fmax(cols(), other.cols())));

        for (int i = 0; i < out.rows(); ++i)
            for (int j = 0; j < out.cols(); ++j) {
                // Perform addition only if the indices are within the dimensions of A and B
                if (i < rows() && j < cols())
                    out(i, j) += (*this)(i, j);
                if (i < other.rows() && j < other.cols())
                    out(i, j) += other(i, j);
            }

        return out;
    }

    Matrix Matrix::operator-(const Matrix &other) const {
        Matrix out(rows(), cols());

        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j) {
                Scalar a = (*this)(i, j);
                Scalar b = (i < other.rows() && j < other.cols()) ? other(i, j) : 0.0;
                out(i, j) = a - b;
            }

        return out;
    }

    Matrix Matrix::operator+(const Scalar &other) const {
        Matrix out(rows(), cols());

        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t j = 0; j < cols(); ++j)
                out(i, j) = (*this)(i, j) + other;

        return out;
    }

    Matrix Matrix::operator-(const Scalar &other) const {
        Matrix out(rows(), cols());

        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t j = 0; j < cols(); ++j)
                out(i, j) = (*this)(i, j) - other;

        return out;
    }

    Matrix Matrix::operator/(const Scalar &other) const {
        Matrix out(rows(), cols());

        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t j = 0; j < cols(); ++j)
                out(i, j) = (*this)(i, j) / other;

        return out;
    }

    Matrix Matrix::operator*(const Scalar &other) const {
        Matrix out(rows(), cols());

        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t j = 0; j < cols(); ++j)
                out(i, j) = (*this)(i, j) * other;

        return out;
    }

    std::ostream &operator<<(std::ostream &os, const Matrix &m) {
        os << std::scientific << std::setprecision(16);
        for (std::size_t row = 0; row < m.rows(); ++row) {
            for (std::size_t col = 0; col < m.cols(); ++col) {
                os << std::setw(23) << m(row, col) << " ";
            }
            os << "\n";
        }
        return os;
    }
} // namespace perceptron::blas
