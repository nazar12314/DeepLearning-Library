//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_INITIALIZER_H
#define NEURALIB_INITIALIZER_H

#include "unsupported/Eigen/CXX11/Tensor"
#include <random>

using Eigen::Tensor;

template<class T>
class Initializer {
protected:
    std::uint32_t seed = std::random_device()();

public:

    virtual Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) = 0;

    virtual Tensor<T, 3, Eigen::RowMajor> get_weights_3d(size_t n_in, size_t n_hidden1, size_t n_hidden2) = 0;

    virtual ~Initializer() = default;

    std::uint32_t get_seed(){return seed;}

    void set_seed(std::uint32_t seed_) {
        seed = seed_;
    }
};

namespace initializers {

    template<class T>
    class RandomNormal: public Initializer<T> {
        double mean;
        double sd;

    public:
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

        Tensor<T, 3, Eigen::RowMajor> get_weights_3d(size_t n_in, size_t n_hidden1, size_t n_hidden2) override {
            Tensor<T, 3, Eigen::RowMajor> weights (n_hidden1, n_hidden2, n_in);
            srand(this -> seed);
            weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

            weights = weights.constant(sd) * weights + weights.constant(mean);

            return weights;
        };
    };

    template<class T>
    class Constant: public Initializer<T> {
        size_t constant;

    public:
        explicit Constant(size_t constant_):
            constant(constant_){};

        Tensor<T, 2, Eigen::RowMajor> get_weights_2d(size_t n_in, size_t n_hidden) override {
            Tensor<T, 2, Eigen::RowMajor> weights(n_hidden, n_in);

            weights.setConstant(constant);

            return weights;
        };

        Tensor<T, 3, Eigen::RowMajor> get_weights_3d(size_t n_in, size_t n_hidden1, size_t n_hidden2) override {
            Tensor<T, 3, Eigen::RowMajor> weights(n_hidden1, n_hidden2, n_in);

            weights.setConstant(constant);

            return weights;
        };
    };

    template<class T>
    class RandomUniform: public Initializer<T> {
        double a;
        double b;

    public:
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

        Tensor<T, 3, Eigen::RowMajor> get_weights_3d(size_t n_in, size_t n_hidden1, size_t n_hidden2) override {
            Tensor<T, 3, Eigen::RowMajor> weights(n_hidden1, n_hidden2, n_in);

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

        Tensor<T, 3, Eigen::RowMajor> get_weights_3d(size_t n_in, size_t n_hidden1, size_t n_hidden2) override {
            Tensor<T, 3, Eigen::RowMajor> weights(n_hidden1, n_hidden2, n_in);

            srand(this -> seed);
            weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

            weights *= weights.constant(2.0 / (n_in + n_hidden1 + n_hidden2));
            return weights;
        };
    };
}


#endif //NEURALIB_INITIALIZER_H
