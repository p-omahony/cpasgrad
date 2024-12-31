#include "engine.h"
#include <memory>
#include <iostream>
#include <vector>
#include <tuple>

Tensor::Tensor(int rows, int cols, std::vector<float> data){
    m_cols = cols;
    m_rows = rows;

    m_data = data;
}

int Tensor::rows(){
    return m_rows;
}

int Tensor::cols(){
    return m_cols;
}

std::tuple<int, int> Tensor::shape(){
    return std::tuple<int, int> (m_rows, m_cols);
}

std::vector<float> Tensor::get_data(){
    return m_data;
}

std::shared_ptr<Tensor> Tensor::get_grad(){
    return grad;
}

void Tensor::set_grad(std::shared_ptr<Tensor> grad_value){
    grad = std::make_shared<Tensor>(
        grad_value -> m_rows,
        grad_value -> m_cols,
        grad_value -> m_data
    );
}

void Tensor::set_grad_fn(const std::function<void()> &fn) {
    grad_fn = fn;
}

std::function<void()> Tensor::get_gradfn(){
    return grad_fn;
}

void Tensor::print(){
	for(int i = 0; i < m_rows; i++){
        for(int j = 0; j < m_cols; j++){
            std::cout << m_data[i*m_cols+j] << " ";
        }
        std::cout << std::endl;
    }
}

std::shared_ptr<Tensor> Tensor::ones(int rows, int cols){
    std::vector<float> data = {};
    for (int i=0; i<rows*cols; i++){
        data.push_back(1);
    }
    return std::make_shared<Tensor>(rows, cols, data);
}


std::shared_ptr<Tensor> Tensor::transpose(){
    std::vector<float> t_data = {};
    for (int i=0; i<m_cols; i++){
        for (int j=0; j<m_rows; j++){
            t_data.push_back(m_data[j*m_cols+i]);
        }
    }

    return std::make_shared<Tensor>(m_cols, m_rows, t_data);
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2){
    std::shared_ptr<Tensor> out_tensor = t1 * t2;

    t1 -> grad_fn = [t1, t2, out_tensor] {
        t1 -> set_grad(out_tensor -> get_grad() * t2 -> transpose());
    };

    t2 -> grad_fn = [t1, t2, out_tensor] {
        t2 -> set_grad(t1 -> transpose() * out_tensor -> get_grad());
    };

    // out_tensor -> set_gradfn("*");
    return out_tensor;
}

std::shared_ptr<Tensor> Tensor::sum(std::shared_ptr<Tensor> t){
    std::vector<float> data = {};
    float sum = 0;
    for(int i=0; i<t -> m_rows*t -> m_cols; i++){
        sum += t -> m_data[i];
    }
    data.push_back(sum);

    t -> grad_fn = [t] {
        t -> set_grad(ones(t -> m_rows, t -> m_cols));
    };

    return std::make_shared<Tensor> (1, 1, data);
}


std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> t1, const std::shared_ptr<Tensor> t2){
    std::vector<float> data = {};
    for(int i=0; i < t1 -> rows(); i++){
        for(int j=0; j < t2 -> cols(); j++){
            float sum = 0;
            for(int k=0; k < t1 ->cols(); k++){
                sum += t1 -> get_data()[i*t1 -> rows()+k] * t2 -> get_data()[k*t2 -> cols()+j];
            }
            data.push_back(sum);
        }
    }

    std::shared_ptr<Tensor> out_tensor = std::make_shared<Tensor>(t1 -> rows(), t2 -> cols(), data);

    return out_tensor;
}
