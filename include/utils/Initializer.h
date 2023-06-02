//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_INITIALIZER_H
#define NEURALIB_INITIALIZER_H

#include "unsupported/Eigen/CXX11/Tensor"
#include <random>

using Eigen::Tensor;

/**
 * @class Initializer
 * @brief Base class for weight and bias initialization.
 * @tparam T The data type of the weights and biases.
 */
template<class T>
class Initializer {
protected:
    std::uint32_t seed = std::random_device()();

public:
    /**
     * @brief Get the 2D weights tensor.
     * @param n_in The number of input units.
     * @param n_hidden The number of hidden units.
     * @return The initialized 2D weights tensor.
     */
    virtual Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) = 0;

    /**
     * @brief Get the 4D weights tensor.
     * @param kernel_size The size of the convolutional kernel.
     * @param n_in The number of input channels.
     * @param n_out The number of output channels.
     * @return The initialized 4D weights tensor.
     */
    virtual Tensor<T, 4, Eigen::RowMajor> get_weights_4d(size_t kernel_size, size_t n_in, size_t n_out) = 0;

    virtual ~Initializer() = default;

    /**
     * @brief Get the seed value used for random initialization.
     * @return The seed value.
     */
    std::uint32_t get_seed(){return seed;}

    /**
     * @brief Set the seed value for random initialization.
     * @param seed_ The seed value.
     */
    void set_seed(std::uint32_t seed_) {
        seed = seed_;
    }

    /**
     * @brief Get the biases tensor.
     * @param n_in The number of input units.
     * @param n_out The number of output units.
     * @param kernel_depth The depth of the convolutional kernel.
     * @return The initialized biases tensor.
     */
    Tensor<T, 3, Eigen::RowMajor> get_biases_3d(size_t n_in, size_t n_out, size_t kernel_depth) {
        Tensor<T, 3, Eigen::RowMajor> biases (n_in, n_out, kernel_depth);
        biases.setConstant(0);

        return biases;
    }
};

namespace initializers {
    /**
     * @class RandomNormal
     * @brief Random normal initializer.
     * @tparam T The data type of the weights and biases.
     */
    template<class T>
    class RandomNormal: public Initializer<T> {
        double mean;
        double sd;

    public:
        /**
         * @brief Construct a RandomNormal initializer.
         * @param mean_ The mean of the normal distribution.
         * @param sd_ The standard deviation of the normal distribution.
         */
        explicit RandomNormal(double mean_ = 0.0, double sd_ = 1.0):
            mean(mean_),
            sd(sd_) {
        };

        Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) override {
            Tensor<T, 2, Eigen::RowMajor> weights (n_hidden, n_in);
            srand(this -> seed);
            weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

            weights = weights.constant(sd) * weights + weights.constant(mean);

            return weights;
        };

        Tensor<T, 4, Eigen::RowMajor> get_weights_4d(size_t kernel_size, size_t n_in, size_t n_out) override {
            Tensor<T, 4, Eigen::RowMajor> weights (kernel_size, kernel_size, n_in, n_out);
            srand(this -> seed);
            weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

            weights = weights.constant(sd) * weights + weights.constant(mean);

            return weights;
        };
    };

    /**
     * @class Constant
     * @brief Constant initializer.
     * @tparam T The data type of the weights and biases.
     */
    template<class T>
    class Constant: public Initializer<T> {
        size_t constant;

    public:
        /**
         * @brief Construct a Constant initializer.
         * @param constant_ The constant value.
         */
        explicit Constant(size_t constant_):
            constant(constant_){};

        Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) override {
            Tensor<T, 2, Eigen::RowMajor> weights(n_hidden, n_in);

            weights.setConstant(constant);

            return weights;
        };

        Tensor<T, 4, Eigen::RowMajor> get_weights_4d(size_t kernel_size, size_t n_in, size_t n_out) override {
            Tensor<T, 4, Eigen::RowMajor> weights(kernel_size, kernel_size, n_in, n_out);

            weights.setConstant(constant);

            return weights;
        };
    };

    /**
     * @class RandomUniform
     * @brief Random uniform initializer.
     * @tparam T The data type of the weights and biases.
     */
    template<class T>
    class RandomUniform: public Initializer<T> {
        double a;
        double b;

    public:
        /**
          * @brief Construct a RandomUniform initializer.
          * @param a_ The lower bound of the uniform distribution.
          * @param b_ The upper bound of the uniform distribution.
          */
        explicit RandomUniform(double a_ = 0, double b_ = 1):
                a(a_),
                b(b_){};

        Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) override {
            Tensor<T, 2, Eigen::RowMajor> weights(n_hidden, n_in);

            srand(this -> seed);
            weights.template setRandom<Eigen::internal::UniformRandomGenerator<double>>();

            Tensor<double, 0> min_w = weights.minimum();
            Tensor<double, 0> max_w = weights.maximum();

            weights =
                    (weights - weights.constant(min_w(0)))
                    /
                    weights.constant(max_w(0) - min_w(0))
                    *
                    weights.constant(b - a)
                    +
                    weights.constant(a);

            return weights;
        };

        Tensor<T, 4, Eigen::RowMajor> get_weights_4d(size_t kernel_size, size_t n_in, size_t n_out) override {
            Tensor<T, 4, Eigen::RowMajor> weights(kernel_size, kernel_size, n_in, n_out);

            srand(this -> seed);
            weights.template setRandom<Eigen::internal::UniformRandomGenerator<double>>();

            Tensor<double, 0> min_w = weights.minimum();
            Tensor<double, 0> max_w = weights.maximum();

            weights =
                    (weights - weights.constant(min_w(0)))
                    /
                    weights.constant(max_w(0) - min_w(0))
                    *
                    weights.constant(b - a)
                    +
                    weights.constant(a);

            return weights;
        };
    };

    /**
     * @class GlorotNormal
     * @brief Glorot normal initializer.
     * @tparam T The data type of the weights and biases.
     */
    template<class T>
    class GlorotNormal: public Initializer<T> {

    public:
        Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) override {
            Tensor<T, 2, Eigen::RowMajor> weights(n_hidden, n_in);

            srand(this -> seed);
            weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

            weights *= weights.constant(2.0 / (n_in + n_hidden)).sqrt();
            return weights;
        };

        Tensor<T, 4, Eigen::RowMajor> get_weights_4d(size_t kernel_size, size_t n_in, size_t n_out) override {
            Tensor<T, 4, Eigen::RowMajor> weights(kernel_size, kernel_size, n_in, n_out);

            srand(this -> seed);
            weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

            weights *= weights.constant(2.0 / (n_in + n_out + kernel_size * 2));
            return weights;
        };
    };
}


#endif //NEURALIB_INITIALIZER_H
