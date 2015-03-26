
#include <stdio.h>

#include "defs.h"
#include "common.h"

#define MATRIX_SIZE 500

void serialMM(int sz, double **A, double **B, double **C)
{
   int i = 0, j = 0, k = 0;

   for (i = 0; i < sz; i++) {
     for (j = 0; j < sz; j++) {
       for (k = 0; k < sz ; k++) {
         C[i][j] = C[i][j] + A[i][k] * B[k][j];
       }
     }
   }

   return;
}

__global__ void MatrixMulKernel (float *Md, float *Nd, float *Pd, int Width)
{
// fill in your CUDA kernel code here	
  1.        __shared__ float Mds [TILE_WIDTH] [TILE_WIDTH];
  2.        __shared__ float Nds [TILE_WIDTH] [TILE_WIDTH];
  3.        int bx = blockIdx.x; int by = blockIdx.y;
  4.        int tx = threadIdx.x; int ty = threadIdx.y;
            //Identify the row and column of the Pd element to work on
  5.        int Row = by * TILE_WIDTH + ty;
  6.        int Col = bx * TILE_WIDTH + tx;
  7.        float Pvalue = 0;
            // Loop over the Md and Nd tiles required to compute the Pd element
  8.        for (int m=0; m < Width / TILE_WIDTH; ++m) {
                // Collaborative (parallel) loading of Md and Nd tiles into shared memory
  9.            Mds [ty] [tx] = Md [Row*Width + (m*TILE_WIDTH + tx)];
  10.           Nds [ty] [tx] = Nd [(m*TILE_WIDTH + ty)*Width + Col];
  11.           __syncthreads(); // make sure all threads have completed copy before calculation
  12.           for (int k = 0; k < TILE_WIDTH; ++k) // Update Pvalue for TKxTK tiles in Mds and Nds
  13.               Pvalue += Mds [ty] [k] * Nds [k] [tx];
  14.           __syncthreads(); // make sure calculation complete before copying next tile
} // m loop
  15.       Pd [Row*Width + Col]  =  Pvalue;

}

void cudaMM(int sz, double **A, double **B, double **E)
{
  	float *d_A, *d_B, *d_E;
    
    int size = sz * sz * sizeof(double);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_E, size);
    
    // copy A and B to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    vecAddKernel<<<ceil(sz * sz/256.0), 256>>>(d_A, d_B, d_E, n);
    
    // copy E from device memory
    cudaMemcpy(E, d_E, size, cudaMemcpyDeviceToHost);
    
    // free device vectors
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_E);
}

int main(int argc, char *argv[])
{
  double **A = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);
  double **B = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);
  double **C = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);
  double **D = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);

  // fill Symmetric matrix A
  for (int i = 0; i < MATRIX_SIZE; i++) {
     for (int j = 0; j <= i; j ++) {
       A[i][j] = (double)min(i+1,j+1);
     }
  } 

  // fill Tridiagonal matrix B
  for (int i = 0; i < MATRIX_SIZE; i++)
    B[i][i] = 1;
  for (int i = 0; i < MATRIX_SIZE - 1; i++)
    B[i][i+1] = 2;
  for (int i = 1; i < MATRIX_SIZE; i++)
    B[i][i-1] = 2;

  double elapse = 0;

  for (int i = 0; i < NUM_RUNS; i++) {
    double seconds = read_timer();
    serialMM(MATRIX_SIZE, A, B, C);
    elapse += (read_timer() - seconds);
  }
  elapse /= NUM_RUNS; 
  printf("serial matrix multiplication: %lf\n", elapse);

  elapse = 0;
  for (int i = 0; i < NUM_RUNS; i++) {

    double seconds = read_timer();
    cudaMM(MATRIX_SIZE, A, B, D);
    elapse += (read_timer() - seconds);
  }
  elapse /= NUM_RUNS; 
  printf("GPU matrix multiplication: %lf\n", elapse);

  if (matrix_compare(N, C, D) != 0) {
     printf("*** error ***\n");
     exit(-1);
  }

  return 0;
}
