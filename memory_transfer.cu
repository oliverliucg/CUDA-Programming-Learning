#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void mem_trs_test(int* input) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("blockIdx.x: %d, threadIdx.x: %d, gid: %d, value: %d\n", blockIdx.x,
	  		 threadIdx.x, gid, input[gid]);
}

__global__ void mem_trs_test(int* input, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    printf("blockIdx.x: %d, threadIdx.x: %d, gid: %d, value: %d\n", blockIdx.x,
           threadIdx.x, gid, input[gid]);
  }
}

__global__ void mem_trs_3d_grid_3d_block(int* input, int size) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  int num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  int bid =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  int gid = tid + bid * num_threads_per_block;
  if (gid < size) {
    // print tid, bid, gid, and value
    printf("tid: %d, bid: %d, gid: %d, value: %d\n", tid, bid, gid, input[gid]);
  }
}

//int main() {
//  int size = 64;
//  int byte_size = sizeof(int) * size;
//  int* h_data = (int*)malloc(byte_size);
//  time_t t;
//  srand((unsigned)time(&t));
//  for (int i = 0; i < size; ++i) {
//    h_data[i] = (int)(rand() & 0xff);
//  }
//  int* d_data;
//  cudaMalloc((void**)&d_data, byte_size);
//  cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice);
//  dim3 block(2, 2, 2);
//  dim3 grid(4 / block.x, 4 / block.y, 4 / block.z);
//  mem_trs_3d_grid_3d_block<<<grid, block>>>(d_data, size);
//  cudaDeviceSynchronize();
//  cudaFree(d_data);
//  free(h_data);
//  cudaDeviceReset();
//  return 0;
//}