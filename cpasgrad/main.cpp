#include "./autograd/engine.cpp"
#include "./autograd/engine.h"
#include <iostream>
#include <memory>
#include <vector>

template <typename T>
void print(const T& value){
    std::cout << value << std::endl;
}

int main(){
    std::vector<float> data_a= {1, 2, 3, 4};
    std::vector<float> data_b= {1, 2, 3, 4, 5, 6};

	std::shared_ptr<Tensor> A = std::make_shared<Tensor>(2, 2, data_a);
	std::shared_ptr<Tensor> B = std::make_shared<Tensor>(2, 3, data_b);

	std::cout << "A: " << std::endl; A -> print(); std::cout << std::endl;
	std::cout << "A.T: " << std::endl; A -> transpose() -> print(); std::cout << std::endl;
	std::cout << "B: " << std::endl; B -> print(); std::cout << std::endl;
	std::cout << "B.T: " << std::endl; B -> transpose() -> print(); std::cout << std::endl;

	std::shared_ptr<Tensor> C = Tensor::matmul(A, B);
	std::cout << "C: " << std::endl; C -> print(); std::cout << std::endl;

	std::shared_ptr<Tensor> loss = Tensor::sum(C);

	loss -> print(); std::cout << std::endl;

	std::function<void()> f = C -> get_gradfn();
	f();
	C -> get_grad() -> print(); std::cout << std::endl;

	std::function<void()> g = A -> get_gradfn();
	g();
	A -> get_grad() -> print(); std::cout << std::endl;

	std::function<void()> d = B -> get_gradfn();
	d();
	B -> get_grad() -> print(); std::cout << std::endl;

    return 0;
}
