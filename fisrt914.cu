#include <stdio.h>
#include <ctime>

__global__ void hello(float f){
    printf("Hello from block %d, thread %d, f=%f\n", blockIdx.x, threadIdx.x, f);
}

__shared__

std::clock_t start=std::clock();
