#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define ARRAY_SIZE 256

#define MAX(x,y) \
       ({ typeof (x) _x = (x); \
           typeof (y) _y = (y); \
         _x > _y ? _x : _y; })

#define MIN(x,y) \
       ({ typeof (x) _x = (x); \
           typeof (y) _y = (y); \
         _x < _y ? _x : _y; })

void compute() {

unsigned iter=0;
float err=1.0;

float tol = 0.000001;
unsigned iter_max = 1000000;

size_t i;
size_t j;

unsigned m = ARRAY_SIZE;
unsigned n = ARRAY_SIZE;
float arr_size = ARRAY_SIZE;


float *A = malloc(sizeof(float) * ARRAY_SIZE * ARRAY_SIZE);
float *Anew = malloc(sizeof(float) * ARRAY_SIZE * ARRAY_SIZE);

//for(size_t i=0;i<ARRAY_SIZE*ARRAY_SIZE;i++) {
//     A[i]=rand();
//  }
// START POINTS:
//  20 === 30
//  ||     ||
//  10 === 20
float STEP_SIZE = 10 / (arr_size-1);
for(i=0; i<ARRAY_SIZE; i++) {
  // upper line
  A[i] = 20 + i*STEP_SIZE;
  // down line
  A[ARRAY_SIZE*(ARRAY_SIZE-1) + i] = 10 + i*STEP_SIZE;
  // left line
  A[i*ARRAY_SIZE] = 20 - i*STEP_SIZE;
  // right line
  A[i*ARRAY_SIZE + (ARRAY_SIZE-1)] = 30 - i*STEP_SIZE;
}

for(unsigned loop = 0; loop < ARRAY_SIZE*ARRAY_SIZE; loop++)
      printf("%f ", A[loop]);

#pragma acc data copy(A[:ARRAY_SIZE * ARRAY_SIZE]) create(Anew[:ARRAY_SIZE * ARRAY_SIZE])

while ( err > tol && iter < iter_max ) {
  
  err = 0.0;
  iter = iter +1;
 
  #pragma acc kernels loop independent collapse(2) reduction(max:err)
  for (j=1; j <m-1; ++j) {
    
    for (i=1; i <n-1; ++i) {
      Anew[i+j*n] = .25 *( A[(i+1)+j*n] + A[(i-1)+j*n] + A[i+(j-1)*n] + A[i+(j+1)*n]);
      err = MAX(err, Anew[i+j*n]-A[i+j*n]);
      }
    }
  
  // cycle for A = Anew:
  #pragma acc kernels loop independent collapse(2)
  for (j=1; j <m-1; ++j) {
    for (i=1; i <n-1; ++i) {
      A[i+j*n] = Anew[i+j*n];
      }
    }
  
  if (iter % 100 == 0) {
      printf("in process iter=%d, err=%e\n", iter, err);  
    }
  }

printf("final iter=%d, err=%e\n", iter, err);
free(A);
free(Anew);

}

int main() {
  struct timeval start, end;
  gettimeofday(&start, NULL);
  compute();
  gettimeofday(&end, NULL);
  float time_spent = (((end.tv_sec + end.tv_usec/1000000) - (start.tv_sec + start.tv_usec/1000000)));
  printf("time spent=%f \n", time_spent);
}