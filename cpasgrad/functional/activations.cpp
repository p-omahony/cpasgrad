#include <memory>
#include "../autograd/engine.h"

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
