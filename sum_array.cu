#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstring>

#include "common.h"

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line,
    bool abort = true) {    
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

__global__ void sum_array_gpu(int* a, int* b, int* c, int* res, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    res[gid] = a[gid] + b[gid] + c[gid];
  }
}

void sum_array_cpu(int* a, int* b, int* c, int* res, int size) {
  for (int i = 0; i < size; ++i) {
    res[i] = a[i] + b[i] + c[i];
  }
}

int main() {
  int size = 1 << 22;
  int block_size = 256;
  int number_of_bytes = size * sizeof(int);

  int *h_a, *h_b, *h_c, *h_res, *d_res, *gpu_res;

  h_a = (int*)malloc(number_of_bytes);
  h_b = (int*)malloc(number_of_bytes);
  h_c = (int*)malloc(number_of_bytes);
  h_res = (int*)malloc(number_of_bytes);
  d_res = (int*)malloc(number_of_bytes);

  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; ++i) {
    h_a[i] = (int)(rand() & 0xff);
    h_b[i] = (int)(rand() & 0xff);
    h_c[i] = (int)(rand() & 0xff);
  }
  clock_t cpu_start, cpu_end;
  cpu_start = clock();
  sum_array_cpu(h_a, h_b, h_c, h_res, size);
  cpu_end = clock();

  memset(d_res, 0, number_of_bytes);

  int *d_a, *d_b, *d_c;
  gpuErrchk(cudaMalloc((void**)&d_a, number_of_bytes));
  gpuErrchk(cudaMalloc((void**)&d_b, number_of_bytes));
  gpuErrchk(cudaMalloc((void**)&d_c, number_of_bytes));
  gpuErrchk(cudaMalloc((void**)&gpu_res, number_of_bytes));

  clock_t htod_start, htod_end;
  htod_start = clock();
  gpuErrchk(cudaMemcpy(d_a, h_a, number_of_bytes, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b, number_of_bytes, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_c, h_c, number_of_bytes, cudaMemcpyHostToDevice));

  htod_end = clock();

  dim3 block(block_size);
  dim3 grid(size / block_size + 1);

  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, gpu_res, size);
  gpu_end = clock();

  gpuErrchk(cudaDeviceSynchronize());

  clock_t dtoh_start, dtoh_end;
  dtoh_start = clock();
  gpuErrchk(cudaMemcpy(d_res, gpu_res, number_of_bytes, cudaMemcpyDeviceToHost));
  dtoh_end = clock();

  // Compare the results
  compareArrays(h_res, d_res, size);

  printf("CPU Execution Time: %4.6f\n",
         (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

  printf("GPU Execution Time: %4.6f\n",
                 (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

  printf("Host to Device Transfer Time: %4.6f\n", 
         (double)((double)(htod_end - htod_start) /
                                 CLOCKS_PER_SEC));
  printf("Device to Host Transfer Time: %4.6f\n",
    (double)((double)(dtoh_end - dtoh_start) /
                CLOCKS_PER_SEC));

  printf("GPU total execution time: %4.6f\n",
		 (double)((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(gpu_res);

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_res);
  free(d_res);
}