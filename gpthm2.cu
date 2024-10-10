// Tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cuda_runtime.h>

class Tensor {
public:
    // 构造函数，接受形状和设备类型（"cpu"或"gpu"）
    Tensor(const std::vector<int>& shape, const std::string& device);
    // 析构函数
    ~Tensor();

    // 复制到CPU
    Tensor cpu() const;
    // 复制到GPU
    Tensor gpu() const;

    // 数据指针
    float* data() const{
        return data_;
    };

    // 获取形状
    std::vector<int> shape() const;

private:
    float* data_;
    std::vector<int> shape_;
    std::string device_;
    size_t size_; // 元素总数
};

#endif // TENSOR_H

//第三步：实现构造函数和析构函数

// Tensor.cpp
#include "Tensor.h"
#include <cstring>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, const std::string& device)
    : shape_(shape), device_(device) {
    // 计算总元素数量
    size_ = 1;
    for (int dim : shape_) {
        size_ *= dim;
    }

    if (device_ == "cpu") {
        // 在CPU上分配内存
        data_ = new float[size_];
    } else if (device_ == "gpu") {
        // 在GPU上分配内存
        cudaError_t err = cudaMalloc(&data_, size_ * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA内存分配失败");
        }
    } else {
        throw std::invalid_argument("未知的设备类型");
    }
}

Tensor::~Tensor() {
    if (device_ == "cpu") {
        delete[] data_;
    } else if (device_ == "gpu") {
        cudaFree(data_);
    }
}


//第四步：实现cpu()和gpu()成员函数

Tensor Tensor::cpu() const {
    if (device_ == "cpu") {
        // 已经在CPU上，直接返回自身的副本
        Tensor tensor(shape_, "cpu");
        std::memcpy(tensor.data_, data_, size_ * sizeof(float));
        return tensor;
    } else {
        // 从GPU复制到CPU
        Tensor tensor(shape_, "cpu");
        cudaMemcpy(tensor.data_, data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        return tensor;
    }
}

Tensor Tensor::gpu() const {
    if (device_ == "gpu") {
        // 已经在GPU上，直接返回自身的副本
        Tensor tensor(shape_, "gpu");
        cudaMemcpy(tensor.data_, data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
        return tensor;
    } else {
        // 从CPU复制到GPU
        Tensor tensor(shape_, "gpu");
        cudaMemcpy(tensor.data_, data_, size_ * sizeof(float), cudaMemcpyHostToDevice);
        return tensor;
    }
}

//第五步：实现ReLU的前向和反向传播

// ReLU.cu
#include "Tensor.h"

__global__ void relu_forward(float* input, float* output, size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void ReLUForward(const Tensor& input, Tensor& output) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("输入和输出的形状必须相同");
    }
    size_t size = input.size();

    if (input.device() == "gpu") {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_forward<<<blocks, threads>>>(input.data(), output.data(), size);
        cudaDeviceSynchronize();
    } else {
        // CPU实现
        float* in_data = input.data();
        float* out_data = output.data();
        for (size_t i = 0; i < size; ++i) {
            out_data[i] = std::max(0.0f, in_data[i]);
        }
    }
}

__global__ void relu_backward(float* grad_output, float* input, float* grad_input, size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
    }
}

void ReLUBackward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
    if (input.shape() != grad_input.shape() || input.shape() != grad_output.shape()) {
        throw std::invalid_argument("输入和输出的形状必须相同");
    }
    size_t size = input.size();

    if (input.device() == "gpu") {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_backward<<<blocks, threads>>>(grad_output.data(), input.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    } else {
        // CPU实现
        float* grad_out_data = grad_output.data();
        float* in_data = input.data();
        float* grad_in_data = grad_input.data();
        for (size_t i = 0; i < size; ++i) {
            grad_in_data[i] = in_data[i] > 0 ? grad_out_data[i] : 0.0f;
        }
    }
}


//第六步：实现Sigmoid的前向和反向传播
//1. 前向传播

__global__ void sigmoid_forward(float* input, float* output, size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

void SigmoidForward(const Tensor& input, Tensor& output) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("输入和输出的形状必须相同");
    }
    size_t size = input.size();

    if (input.device() == "gpu") {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sigmoid_forward<<<blocks, threads>>>(input.data(), output.data(), size);
        cudaDeviceSynchronize();
    } else {
        // CPU实现
        float* in_data = input.data();
        float* out_data = output.data();
        for (size_t i = 0; i < size; ++i) {
            out_data[i] = 1.0f / (1.0f + std::exp(-in_data[i]));
        }
    }
}

//2. 反向传播

__global__ void sigmoid_backward(float* grad_output, float* output, float* grad_input, size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
    }
}

void SigmoidBackward(const Tensor& grad_output, const Tensor& output, Tensor& grad_input) {
    if (output.shape() != grad_input.shape() || output.shape() != grad_output.shape()) {
        throw std::invalid_argument("输入和输出的形状必须相同");
    }
    size_t size = output.size();

    if (output.device() == "gpu") {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sigmoid_backward<<<blocks, threads>>>(grad_output.data(), output.data(), grad_input.data(), size);
        cudaDeviceSynchronize();
    } else {
        // CPU实现
        float* grad_out_data = grad_output.data();
        float* out_data = output.data();
        float* grad_in_data = grad_input.data();
        for (size_t i = 0; i < size; ++i) {
            grad_in_data[i] = grad_out_data[i] * out_data[i] * (1.0f - out_data[i]);
        }
    }
}
// Ops.h
#ifndef OPS_H
#define OPS_H

#include "Tensor.h"

// ReLU前向和反向传播
void ReLUForward(const Tensor& input, Tensor& output);
void ReLUBackward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input);

// Sigmoid前向和反向传播
void SigmoidForward(const Tensor& input, Tensor& output);
void SigmoidBackward(const Tensor& grad_output, const Tensor& output, Tensor& grad_input);

#endif // OPS_H


// main.cpp
#include <iostream>
#include "Tensor.h"
#include "Ops.h"

int main() {
    // 定义形状
    std::vector<int> shape = {1024};

    // 在CPU上创建Tensor
    Tensor input(shape, "cpu");
    Tensor output(shape, "cpu");

    // 初始化输入数据
    float* data = input.data();
    for (size_t i = 0; i < input.size(); ++i) {
        data[i] = i - 512; // 从-512到511
    }

    // 将数据移动到GPU
    Tensor input_gpu = input.gpu();
    Tensor output_gpu(shape, "gpu");

    // 执行ReLU前向传播
    ReLUForward(input_gpu, output_gpu);

    // 将结果移动回CPU
    output = output_gpu.cpu();

    // 打印部分结果
    float* out_data = output.data();
    for (int i = 510; i < 515; ++i) {
        std::cout << "input: " << data[i] << ", output: " << out_data[i] << std::endl;
    }

    return 0;
}