#include <algorithm>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "../autograd/engine.h"

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> t, const float c){
    int rows = t -> rows();
    int cols = t -> cols();
    std::vector<float> t_data = t -> get_data();
    std::vector<float> div_data = {};
    for (int i=0; i < rows*cols; i++){
        div_data.push_back(t_data[i]/c);
    }

    return std::make_shared<Tensor>(rows, cols, div_data);
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> t1, const std::shared_ptr<Tensor> t2){
    int t1_cols = t1 -> cols();
    int t1_rows = t1 -> rows();

    int t2_cols = t2 -> cols();

    std::vector<float> t1_data = t1 -> get_data();
    std::vector<float> t2_data = t2 -> get_data();

    std::vector<float> data = {};
    for(int i=0; i < t1_rows; i++){
        for(int j=0; j < t2_cols; j++){
            float sum = 0;
            for(int k=0; k < t1_cols; k++){
                sum += t1_data[i*t1_rows+k] * t2_data[k*t2_cols+j];
            }
            data.push_back(sum);
        }
    }

    std::shared_ptr<Tensor> out_tensor = std::make_shared<Tensor>(t1_rows, t2_cols, data);

    return out_tensor;
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> t, const float c){
    int rows = t -> rows();
    int cols = t -> cols();
    std::vector<float> t_data = t -> get_data();
    std::vector<float> div_data = {};
    for (int i=0; i < rows*cols; i++){
        div_data.push_back(t_data[i]*c);
    }

    return std::make_shared<Tensor>(rows, cols, div_data);
}

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor> t){
    int rows = t -> rows();
    int cols = t -> cols();
    std::vector<float> t_data = t -> get_data();
    std::vector<float> relu_data = {};
    for (int i=0; i < rows*cols; i++){
        if (t_data[i] > 0.){
            relu_data.push_back(t_data[i]);
        } else {
            relu_data.push_back(0);
        }
    }

    return std::make_shared<Tensor>(rows, cols, relu_data);
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor> t){
    int rows = t -> rows();
    int cols = t -> cols();
    std::vector<float> t_data = t -> get_data();
    std::vector<float> exp_data = {};
    for (int i=0; i < rows*cols; i++){
        exp_data.push_back(std::exp(t_data[i]));
    }

    return std::make_shared<Tensor>(rows, cols, exp_data);
}

std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor> t){
    int rows = t -> rows();
    int cols = t -> cols();
    std::vector<float> t_data = t -> get_data();
    std::vector<float> log_data = {};
    for (int i=0; i < rows*cols; i++){
        log_data.push_back(std::log(t_data[i]));
    }

    return std::make_shared<Tensor>(rows, cols, log_data);
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor> t){
    int rows = t -> rows();
    int cols = t -> cols();
    std::vector<float> t_data = t -> get_data();
    std::vector<float> sum_data = {};
    float sum = 0;
    for (int i=0; i < rows*cols; i++){
        sum += t_data[i];
    }
    sum_data.push_back(sum);

    return std::make_shared<Tensor>(1, 1, sum_data);
}

std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor> t){
    int rows = t -> rows();
    int cols = t -> cols();
    std::shared_ptr<Tensor> exp_t = exp(t);
    std::shared_ptr<Tensor> sum_t = sum(exp_t);
    const float c = sum_t -> get_data()[0];

    return exp_t/c;
}

std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> y_pred, std::shared_ptr<Tensor> y_true){
    std::shared_ptr<Tensor> softmax_t = softmax(y_pred);

    std::shared_ptr<Tensor> log_softmax_t = log(softmax_t);
    std::shared_ptr<Tensor> entropy = log_softmax_t * y_true -> transpose();

    return sum(entropy)*-1.;
}
