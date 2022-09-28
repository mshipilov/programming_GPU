#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000
#define FLOAT_TYPE float
#define TWO_PI (((FLOAT_TYPE) 2.0) * M_PI)
#define RADIAN_DIV (TWO_PI / (FLOAT_TYPE)(ARRAY_SIZE - 1))
#define SIN sin

FLOAT_TYPE
compute_sin() {
  FLOAT_TYPE *sin_array = malloc(sizeof(FLOAT_TYPE) * ARRAY_SIZE);
  # pragma acc kernels
  for (unsigned i = 0; i < ARRAY_SIZE; ++i) {
    sin_array[i] = SIN(i * RADIAN_DIV);
  }

  FLOAT_TYPE sin_sum = (FLOAT_TYPE)0.0;
  # pragma acc kernels
  for (unsigned i = 0; i < ARRAY_SIZE; ++i) {
    sin_sum += sin_array[i];
  }

  free(sin_array);

  return sin_sum;
}

int main() {
  FLOAT_TYPE sin_sum = compute_sin();

  printf("%e\n", sin_sum);
}
