// --*-c++-*--

#include <stdio.h>
#include <stdlib.h>

#include "defs.h"

double** Make2DDoubleArray(int arraySizeX, int arraySizeY)
{
  double** theArray;
  theArray = (double**) malloc(arraySizeX*sizeof(double*));

  for (int i = 0; i < arraySizeX; i++)
    theArray[i] = (double*) malloc(arraySizeY*sizeof(double));

  for (int i = 0; i < arraySizeX; i++)
    for (int j = 0; j < arraySizeX; j++)
       theArray[i][j] = 0;

  return theArray;
}

int matrix_compare(int sz, double **A, double **B)
{
  int error = 0;

  for (int i = 0; i < sz; i++) {
    for (int j = 0; j < sz; j++) {
      if (A[i][j] != B[i][j]) {
        error = 1;
        return error;
      }
    }
  }
  return error;
}

