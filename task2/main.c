#include <stdio.h>
#include <math.h>
#include <time.h>

#define ARRAY_SIZE 128
#define FLOAT_TYPE float

#define MAX(x,y) \
       ({ typeof (x) _x = (x); \
           typeof (y) _y = (y); \
         _x > _y ? _x : _y; })

#define MIN(x,y) \
       ({ typeof (x) _x = (x); \
           typeof (y) _y = (y); \
         _x < _y ? _x : _y; })

void
compute() {
unsigned iter=0;
float err=1.0;

float tol = 0.000001;
unsigned iter_max = 1000000;

unsigned i;
unsigned j;

unsigned m = ARRAY_SIZE;
unsigned n = ARRAY_SIZE;

FLOAT_TYPE *A = malloc(sizeof(FLOAT_TYPE) * ARRAY_SIZE * ARRAY_SIZE);
FLOAT_TYPE *Anew = malloc(sizeof(FLOAT_TYPE) * ARRAY_SIZE * ARRAY_SIZE);
for(int i=1;i<ARRAY_SIZE*ARRAY_SIZE;i++) {
     A[i]=rand();
  }

while ( err > tol && iter < iter_max ) {
  
  err = 0.0;
  iter = iter +1;
 
  # pragma acc kernels
  for (j=1; j <m-1; ++j) {
    for (i=1; i <n-1; ++i) {
      Anew[i+j*n] = .25 *( A[(i+1)+j*n] + A[(i-1)+j*n] + A[i+(j-1)*n] + A[i+(j+1)*n]);
      err = MAX(err, Anew[i+j*n]-A[i+j*n]);
      }
    }

  // add cycle for A = Anew:
  for (j=1; j <m; ++j) {
    for (i=1; i <n; ++i) {
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
  clock_t begin = clock();
  compute();
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("time spent=%d \n", time_spent);
}
