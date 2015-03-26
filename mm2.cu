
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

__global__ void cudakernel(double *buf)
{
  // fill in your CUDA kernel code here
}

void cudaMM(int sz, double **A, double **B, double **E)
{
  // fill in your CUDA MM code here
  // you should store B as efficiently as possible
}

int main(int argc, char *argv[])
{
  double **A = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);
  double **B = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);
  double **C = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);
  double **E = Make2DDoubleArray(MATRIX_SIZE, MATRIX_SIZE);

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
    cudaMM(MATRIX_SIZE, A, B, E);
    elapse += (read_timer() - seconds);
  }
  elapse /= NUM_RUNS; 
  printf("GPU matrix multiplication: %lf\n", elapse);

  if (matrix_compare(N, C, E) != 0) {
     printf("*** error ***\n");
     exit(-1);
  }

  return 0;
}
