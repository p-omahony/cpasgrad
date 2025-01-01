#include <memory>
#include <cmath>
#include "../autograd/engine.h"
#include "./operations.cpp"

std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> y_pred, std::shared_ptr<Tensor> y_true){
    std::shared_ptr<Tensor> softmax_t = softmax(y_pred);

    std::shared_ptr<Tensor> log_softmax_t = log(softmax_t);
    std::shared_ptr<Tensor> entropy = log_softmax_t * y_true -> transpose();

    return sum(entropy)*-1.;
}
