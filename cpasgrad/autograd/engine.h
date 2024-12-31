#ifndef ENGINE_H
#define ENGINE_H

#include <cmath>
#include <csetjmp>
#include <functional>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <tuple>


class Tensor {

  public:
    Tensor(int rows, int cols, std::vector<float> data);

    int rows();
    int cols();
    std::tuple<int, int> shape();

    void print();
    void print_grad();

    std::vector<float> get_data();

    void set_grad(std::shared_ptr<Tensor> grad_value);
    std::shared_ptr<Tensor> get_grad();

    void set_grad_fn(const std::function<void()> &fn);
    std::function<void()> get_gradfn();

    std::shared_ptr<Tensor> transpose();

    static std::shared_ptr<Tensor> ones(int rows, int cols);
    static std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> t);
    static std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);

  private:
    int m_rows;
    int m_cols;

    std::shared_ptr<Tensor> grad;
    std::function<void()> grad_fn;

    std::vector<float> m_data;

    std::shared_ptr<Tensor> prev;

};

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> t1, const std::shared_ptr<Tensor> t2);

#endif