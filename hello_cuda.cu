#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda() { 
	printf("Hello CUDA from GPU!\n"); 
}

//int main() {
//  int nx, ny;
//  nx = 16;
//  ny = 4;
//  dim3 block(8, 2);
//  dim3 grid(nx / block.x, ny / block.y);
//  hello_cuda<<<grid, block>>>();
//  // hello_cuda<<<2, 20>>>();
//  cudaDeviceSynchronize();
//  cudaDeviceReset();
//  return 0;
//}