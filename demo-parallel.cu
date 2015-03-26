// source: http://stackoverflow.com/questions/7663343/simplest-possible-example-to-show-gpu-outperform-cpu-using-cuda

#include <stdio.h>

#include "defs.h"
#include "common.h"

__global__ void cudakernel(double *buf)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  buf[i] = 1.0 * i / N;
  for (int j = 0; j < M; j++)
    buf[i] = buf[i] * buf[i] - 0.25;
}

int main(int argc, char *argv[])
{
  double t[2] = { 0 };
  double data[N] = { 0 } ; 
  double *d_data;

  t[0] = read_timer();

  cudaMalloc(&d_data, N * sizeof(double));
  cudakernel<<<N/256, 256>>>(d_data);

  cudaMemcpy(data, d_data, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_data); 

  t[1] = read_timer();
  
  for (int i = 1; i < M/10; i *= 10) { 
    printf("data[%06d] = %lf\n", i, data[i]);
  }

  printf("total time: %lf\n", t[1] - t[0]);

  return 0;
}
