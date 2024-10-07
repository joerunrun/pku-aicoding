#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
using IntVector = std::vector<int>;
using FloatVector = std::vector<float>;


class Tensor {
public:
    // 构造函数
    Tensor( std::vector<int>& shape, const std::string& device) 
         {
            this->shape_ = shape;
            this->device_ = device;
            this->data_.resize(1);
            for (int dim : shape) {
                this->data_[0] *= dim;
            }
            this->data_.resize(data_[0]);
        if (device_ == "cpu") {
            std::cout << "Tensor on CPU" << std::endl;
        } 
        else {
            float* device_data;
            std::cout << "Tensor on GPU" << std::endl;
            cudaMalloc(&device_data, data_.size() * sizeof(float));
            cudaMemcpy(device_data, data_.data(), data_.size() * sizeof(float), cudaMemcpyHostToDevice);
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
            Tensor* newtensor=new Tensor(this ->shape_, "cpu");
            cudaMemcpy(newtensor, data_.data(), data_.size() * sizeof(int), cudaMemcpyDeviceToHost);
            return newtensor;
        } else {
            std::cout << "Tensor is already on CPU" << std::endl;
        }
    }

    // 切换到GPU
    Tensor*  gpu() {
        if (device_ != "gpu") {
            
            std::cout << "Tensor moved to GPU" << std::endl;
            Tensor* newtensor;
            cudaMalloc(&newtensor, data_.size() * sizeof(float));
            cudaMemcpy(newtensor, data_.data(), data_.size() * sizeof(int), cudaMemcpyHostToDevice);
            newtensor->device_ = "gpu";
            return newtensor;
        } else {
            std::cout << "Tensor is already on GPU" << std::endl;
        }
    }

    // 设置张量的值
    void set_value(const std::vector<int>& indices, int value) {
        int index = get_flat_index(indices);
        if (index >= 0 && index < data_.size()) {
            data_[index] = value;
        } else {
            std::cerr << "Index out of bounds" << std::endl;
        }
    }

    // 获取张量的值
    int get_value(const std::vector<int>& indices) const {
        int index = get_flat_index(indices);
        if (index >= 0 && index < data_.size()) {
            return data_[index];
        } else {
            std::cerr << "Index out of bounds" << std::endl;
            return -1; // 返回一个错误值
        }
    }
    FloatVector relu() {
        
    FloatVector output;
    FloatVector* input=&(this->data_);
    output.resize(input.size());
    float* device_input;
    float* device_output;
    cudaMalloc(&device_input, input.size() * sizeof(float));
    cudaMalloc(&device_output, input.size() * sizeof(float));
    cudaMemcpy(device_input, input, input.size() * sizeof(float), cudaMemcpyHostToDevice);
    relu_kernel<<<(input.size() + 255) / 256, 256>>>(device_input, device_output, input.size());
    cudaMemcpy(output.data(), device_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}
    
    FloatVector sigmoid() {
        FloatVector output;
        FloatVector input=this->data_;
        output.resize(input.size());
        float* device_input;
        float* device_output;
        cudaMalloc(&device_input, input.size() * sizeof(float));
        cudaMalloc(&device_output, input.size() * sizeof(float));
        cudaMemcpy(device_input, input, input.size() * sizeof(float), cudaMemcpyHostToDevice);
        sigmoid_kernel<<<(input.size() + 255) / 256, 256>>>(device_input, device_output, input.size());
        cudaMemcpy(output.data(), device_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_input);
        cudaFree(device_output);
        return output;
    }
    FloatVector relu_backward(FloatVector grad) {
        FloatVector output;
        FloatVector input=this->data_;
        output.resize(input.size());
        float* device_input;
        float* device_output;
        float* device_grad;
        cudaMalloc(&device_input, input.size() * sizeof(float));
        cudaMalloc(&device_output, input.size() * sizeof(float));
        cudaMalloc(&device_grad, grad.size() * sizeof(float));
        cudaMemcpy(device_input, input, input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_grad, grad, grad.size() * sizeof(float), cudaMemcpyHostToDevice);
        relu_backward_kernel<<<(input.size() + 255) / 256, 256>>>(device_input, device_grad, device_output, input.size());
        cudaMemcpy(output.data(), device_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_grad);
        return output;
    }


private:
    std::vector<int> shape_;
    std::string device_;
    std::vector<float> data_; // 用于存储张量的数据
    float* device_data;

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

__global__ void relu_kernel(float* input,float* output,int size){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size){
        output[idx]=input[idx]>0?input[idx]:0;
    }
}


__global__ void sigmoid_kernel(float* input,float* output,int size){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size){
        output[idx]=1/(1+exp(-input[idx]));
    }
}

__global__ void relu_backward_kernel(float* input,float* grad,float* output,int size){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size){
        output[idx]=input[idx]>0?grad[idx]:0;
    }
}


int main() {
    std::vector<int> shape = {2, 3, 4};
    Tensor tensor(shape, "cpu");
    tensor.
    // 设置张量的值
    tensor.set_value({0, 1, 2}, 10);
    tensor.set_value({1, 2, 3}, 20);

    // 获取张量的值
    std::cout << "Value at [0, 1, 2]: " << tensor.get_value({0, 1, 2}) << std::endl;
    std::cout << "Value at [1, 2, 3]: " << tensor.get_value({1, 2, 3}) << std::endl;

    tensor.gpu();
    tensor.cpu();

    return 0;
}