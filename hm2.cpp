#include<iostream>
#include<vector>


class Tensor{
    std::vector<int> shape;
    bool is_cuda;
    float* data;
    public:
        Tensor(std::vector<int> shape, bool is_cuda){
            this->shape = shape;
            this->is_cuda = is_cuda;
            int total_size=1;
            for(int i=0;i<shape.size();i++){
                total_size*=shape[i];
            }
            this->data =new float[total_size];
    }
};
Tensor a(std::vector<int>{2,3}, false);
int main(){
    std::cout<<"Hello World\n";
    return 0;
}