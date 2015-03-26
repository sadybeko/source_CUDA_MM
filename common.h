#ifndef _COMMONS_H
#define _COMMONS_H

extern double read_timer();
extern int matrix_compare(int sz, double **A, double **B);
extern double** Make2DDoubleArray(int arraySizeX, int arraySizeY);
extern void fillMatrix(int size, double** matrix);

#endif 
