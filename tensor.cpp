#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include<vector>
#include<string>
using IntVector = std::vector<int>;
using FloatVector = std::vector<float>;
__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx]>0? input[idx] : 0;
    }
}


__global__ void sigmoid_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1 / (1 + exp(-input[idx]));
    }
}

__global__ void relu_backward_kernel(float* input, float* grad, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? grad[idx] : 0;
    }
}

class Tensor {
public:
    std::vector<int> shape_;
    std::string device_;
    float* data_;
    float* device_data;
    int lenth = 1;
    // 构造函数
    Tensor(std::vector<int>& shape, const std::string& device)
    {
        this->shape_ = shape;
        this->device_ = device;
        
        for (int dim : shape) {
            lenth *= dim;
        }
        data_ = new float(lenth);
        if (device_ == "cpu") {
            std::cout << "Tensor on CPU" << std::endl;
        }
        else {
            float* device_data;
            std::cout << "Tensor on GPU" << std::endl;
            cudaMalloc(&device_data, lenth * sizeof(float));
            cudaMemcpy(device_data, data_, lenth * sizeof(float), cudaMemcpyHostToDevice);
        }

    }
    




    // 析构函数
    ~Tensor() {
        std::cout << "Tensor destroyed" << std::endl;
        if (device_ == "gpu") {
            cudaFree(device_data);
        }
    }

    // 切换到CPU
    Tensor* cpu() {
        if (device_ != "cpu") {
            device_ = "cpu";
            std::cout << "Tensor moved to CPU" << std::endl;
            Tensor* newtensor = new Tensor(this->shape_, "cpu");
            cudaMemcpy(newtensor, data_, lenth * sizeof(int), cudaMemcpyDeviceToHost);
            return newtensor;
        }
        else {
            std::cout << "Tensor is already on CPU" << std::endl;
        }
    }

    // 切换到GPU
    Tensor gpu() {
        if (device_ != "gpu") {

            std::cout << "Tensor moved to GPU" << std::endl;
            Tensor newtensor(this->shape_,"gpu");
            float* datanew = this->data_;
            cudaMalloc(&datanew, lenth * sizeof(float));
            cudaMemcpy(datanew, data_, lenth * sizeof(float), cudaMemcpyHostToDevice);
            newtensor.data_ = datanew;
            return newtensor;
        }
        else {
            std::cout << "Tensor is already on GPU" << std::endl;
        }
    }

    // 设置张量的值
    void set_value(const std::vector<int>& indices, int value) {
        int index = get_flat_index(indices);
        if (index >= 0 && index < lenth) {
            data_[index] = value;
        }
        else {
            std::cerr << "Index out of bounds" << std::endl;
        }
    }

    // 获取张量的值
    int get_value(const std::vector<int>& indices) const {
        int index = get_flat_index(indices);
        if (index >= 0 && index < lenth) {
            return data_[index];
        }
        else {
            std::cerr << "Index out of bounds" << std::endl;
            return -1; // 返回一个错误值
        }
    }
    void relu() {

        float* output=new float(lenth);
        float* input = &(this->data_[0]);
        
        float* device_input;
        float* device_output;
        cudaMalloc(&device_input, lenth * sizeof(float));
        cudaMalloc(&device_output, lenth * sizeof(float));
        cudaMemcpy(device_input, input, lenth * sizeof(float), cudaMemcpyHostToDevice);
        relu_kernel <<<(lenth + 255) / 256, 256 >>> (device_input, device_output,lenth);
        cudaMemcpy(output, device_output, lenth * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_input);
        cudaFree(device_output);
        this->data_ = output;
    }

    void sigmoid() {
        float* output;
        float* input = &(this->data_[0]);

        float* device_input;
        float* device_output;
        cudaMalloc(&device_input, lenth * sizeof(float));
        cudaMalloc(&device_output, lenth * sizeof(float));
        cudaMemcpy(device_input, input, lenth * sizeof(float), cudaMemcpyHostToDevice);
        sigmoid_kernel << <(lenth + 255) / 256, 256 >> > (device_input, device_output, lenth);
        cudaMemcpy(output, device_output, lenth * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_input);
        cudaFree(device_output);
        this->data_ = output;
    }
     Tensor relu_backward(Tensor& grad) {
        float* output;
        float* input = this->data_;
        
        float* device_input;
        float* device_output;
        float* device_grad;
        cudaMalloc(&device_input, lenth * sizeof(float));
        cudaMalloc(&device_output, lenth * sizeof(float));
        cudaMalloc(&device_grad, grad.lenth * sizeof(float));
        cudaMemcpy(device_input, input, lenth * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_grad, grad.data_, grad.lenth * sizeof(float), cudaMemcpyHostToDevice);
        relu_backward_kernel <<<(lenth + 255) / 256, 256 >>> (device_input, device_grad, device_output, lenth);
        cudaMemcpy(output, device_output, lenth * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_grad);
        grad.data_ = output;
        return grad;
    }
    


private:
    // 用于存储张量的数据
    ;

    // 将多维索引转换为一维索引
    int get_flat_index(const std::vector<int>& indices) const {
        if (indices.size() != shape_.size()) {
            std::cerr << "Incorrect number of indices" << std::endl;
            return -1;
        }
        int flat_index = 0;
        int stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                std::cerr << "Index out of bounds" << std::endl;
                return -1;
            }
            flat_index += indices[i] * stride;
            stride *= shape_[i];
        }
        return flat_index;
    }
};




int main() {
    std::vector<int> shape = { 2, 3, 4 };
    Tensor tensor(shape, "cpu");
    tensor.set_value({ 0,1,2 }, 10);
        // 设置张量的值.set_value({ 0, 1, 2 }, 10);
    tensor.set_value({ 1, 2, 3 }, -20);
    tensor.relu();
  
    // 获取张量的值
    std::cout << "Value at [0, 1, 2]: " << tensor.get_value({ 0, 1, 2 }) << std::endl;
    std::cout << "Value at [1, 2, 3]: " << tensor.get_value({ 1, 2, 3 }) << std::endl;

    tensor.gpu();
    tensor.cpu();

    return 0;
}