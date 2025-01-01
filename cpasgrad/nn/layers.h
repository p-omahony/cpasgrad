#include <memory>
#include "../autograd/engine.h"

class Linear {
    private:
        int nin;
        int nout;

        std::shared_ptr<Tensor> W;
        std::shared_ptr<Tensor> b;
    public:
        Linear(int nin, int nout);

        int input_size();
        std::shared_ptr<Tensor> weight();

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> X);
};
