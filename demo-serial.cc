#include <stdio.h>
#include "defs.h"
#include "common.h"

int main(int argc, char *argv[])
{
  double t[2] = { 0 };
  double data[N] = { 0 };

  t[0] = read_timer();

  for(int i = 0; i < N; i++) {
    data[i] = 1.0 * i / N;
    for(int j = 0; j < M; j++) {
      data[i] = data[i] * data[i] - 0.25;
    }
  }

  t[1] = read_timer();

  for (int i = 1; i < M/10; i *= 10) { 
    printf("data[%06d] = %f\n", i, data[i]);
  }

  printf("total time: %lf\n", t[1] - t[0]);

  return 0;
}
