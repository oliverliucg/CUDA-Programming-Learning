#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unqiue_idx_calc_threadIdx(int* input) {
  int tid = threadIdx.x;
  printf("threadIdx: %d, value: %d\n", tid, input[tid]);
}

__global__ void unqiue_gid_calculation(int* input) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * blockDim.x;
  int gid = tid + offset;
  printf("blockIdx.x: %d, threadIdx.x: %d, gid: %d, value: %d\n", blockIdx.x,
         tid, gid, input[gid]);
}

__global__ void unique_gid_calculation_2d(int* input) {
  int tid = threadIdx.x;
  int threadsPerRow = gridDim.x * blockDim.x;
  int rowOffset = blockIdx.y * threadsPerRow;
  int colOffset = blockIdx.x * blockDim.x;
  int gid = tid + rowOffset + colOffset;
  printf(
      "blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, value: %d\n",
      blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

__global__ void unique_gid_calculation_2d_2d(int* input) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int threadPerBlock = blockDim.x * blockDim.y;
  int threadPerRow = gridDim.x * threadPerBlock;
  int rowOffset = blockIdx.y * threadPerRow;
  int colOffset = blockIdx.x * threadPerBlock;
  int gid = tid + rowOffset + colOffset;
  printf(
      "blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d, gid: "
      "%d, value: %d\n",
      blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gid, input[gid]);
}

//int main() {
//  int array_size = 16;
//  int array_byte_size = sizeof(int) * array_size;
//  int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 22, 43, 56, 4, 76, 81, 94, 32};
//
//  for (int i = 0; i < array_size; i++) {
//    printf("%d ", h_data[i]);
//  }
//  printf("\n\n");
//
//  int* d_data;
//  cudaMalloc((void**)&d_data, array_byte_size);
//  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
//
//  dim3 block(2, 2);
//  dim3 grid(2, 2);
//
// /* unqiue_idx_calc_threadIdx<<<grid, block>>>(d_data);*/
//  unique_gid_calculation_2d_2d<<<grid, block>>>(d_data);
//  cudaDeviceSynchronize();
//
//  cudaFree(d_data);
//  cudaDeviceReset();
//  return 0; 
//}