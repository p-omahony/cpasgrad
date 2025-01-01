#include "layers.h"
#include "../autograd/engine.h"
#include "../functional/losses.cpp"

Linear::Linear(int input_size, int output_size){
    nin = input_size;
    nout = output_size;

    W = Tensor::randn(input_size, output_size);
    b = Tensor::randn(input_size, input_size);
}

int Linear::input_size(){
    return nin;
}

std::shared_ptr<Tensor> Linear::weight(){
    return W;
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> X){
    return X*W;
}
