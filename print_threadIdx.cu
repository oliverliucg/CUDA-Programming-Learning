#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIdx() {
  printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x,
         threadIdx.y, threadIdx.z);
}

__global__ void print_blockIdx_and_gridIdx() {
  printf(
      "blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, gridDim.x: %d, "
      "gridDim.y: %d, gridDim.z: %d\n",
      blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
}

//int main() {
//  int nx, ny, nz;
//  nx = 4;
//  ny = 4;
//  nz = 4;
//  dim3 block(2, 2, 2);
//  dim3 grid(nx / block.x, ny / block.y);
// /* print_threadIdx<<<grid, block>>>();*/
//  print_blockIdx_and_gridIdx<<<grid, block>>>();
//  cudaDeviceSynchronize();
//  cudaDeviceReset();
//  return 0;
//}