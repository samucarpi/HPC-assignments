#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "cholesky.h"

/* Array initialization. */

#ifdef INIT_DEBUG
  static void init_array(int n,DATA_TYPE POLYBENCH_1D(p, N, n),DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
  {
    int i, j;
    for (i = 0; i < n; i++)
      p[i] = 0.0;

    FILE *fp = fopen("custom_matrix.txt", "r");
    if (fp != NULL) {
      for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
          fscanf(fp, "%lf", &A[i][j]);
      fclose(fp);
    }else{
      exit(1);
    }
  }
#else
  static void init_array(int n,DATA_TYPE POLYBENCH_1D(p, N, n),DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
  {
    int i, j;

    for (i = 0; i < n; i++)
    {
      p[i] = 1.0 / n;
      for (j = 0; j < n; j++)
        A[i][j] = 1.0 / n;
    }
  }
#endif
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,DATA_TYPE POLYBENCH_2D(A, N, N, n, n),DATA_TYPE POLYBENCH_1D(p, N, n))
{
  int i, j;

  fprintf(stderr, "\nMATRICE RISULTANTE:\n");
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if (j < n - 1)
        fprintf(stderr, " ");
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\nVALORI DIAGONALI:\n");
  for (i = 0; i < n; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, 1/p[i]);
    if (i < n - 1)
      fprintf(stderr, " ");
  }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return.
  static void kernel_cholesky(int n,DATA_TYPE POLYBENCH_1D(p, N, n),DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
  {
    int i, j, k;
    DATA_TYPE x;
    for (i = 0; i < _PB_N; ++i)
    {
      x = A[i][i];
      for (j = 0; j <= i - 1; ++j){
        x -= A[i][j] * A[i][j];
      }
      p[i] = 1.0 / sqrt(x);
      for (j = i + 1; j < _PB_N; ++j)
      {
        x = A[i][j];
        for (k = 0; k <= i - 1; ++k)
          x -= A[j][k] * A[i][k];
        A[j][i] = x * p[i];
      }
    }
  }
*/
#define BLOCK_SIZE (512)
#ifdef SAMUELE

__global__ void kernel_cholesky_device(DATA_TYPE *p, DATA_TYPE *A, int i, int n)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j>i && j < n){
    DATA_TYPE x = A[i * n + j];
    for (int k = 0; k <= i - 1; ++k)
      x -= A[j * n + k] * A[i * n + k];
    A[j * n + i] = x * p[i];
  }
}

static void kernel_cholesky(int n,DATA_TYPE POLYBENCH_1D(p, N, n),DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){
  int i, j, k;
  DATA_TYPE x;
  DATA_TYPE *d_p, *d_A;
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  cudaMalloc((void **)&d_p, n * sizeof(DATA_TYPE));
  cudaMalloc((void **)&d_A, n * n * sizeof(DATA_TYPE));
  cudaMemcpy(d_A, &A[0][0], n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  for (i = 0; i < _PB_N; ++i)
  {
    x = A[i][i];
    for (j = 0; j <= i - 1; ++j){
      x -= A[i][j] * A[i][j];
    }
    p[i] = 1.0 / sqrt(x);
    cudaMemcpy(&d_p[i], &p[i], sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    kernel_cholesky_device<<<dimGrid, dimBlock>>>(d_p, d_A, i, n);
    cudaDeviceSynchronize();
    for (int j = i + 1; j < n; ++j){
      cudaMemcpy(&A[j][i], &d_A[j * n + i], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    }
  }
  cudaFree(d_p);
  cudaFree(d_A);
}
#endif

#ifdef MATTIA
#define IDX(i,j,n) ((i)*(n)+(j))

__global__ void kernel_cholesky(int n, DATA_TYPE *d_p, DATA_TYPE *d_A){
  // Algoritmo sequenziale eseguito da un singolo thread
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int i, j, k;
    DATA_TYPE x;
    
    for (i = 0; i < n; ++i) {
      // Calcola elemento diagonale
      x = d_A[IDX(i,i,n)];
      for (j = 0; j <= i - 1; ++j) {
        x -= d_A[IDX(i,j,n)] * d_A[IDX(i,j,n)];
      }
      d_p[i] = 1.0 / sqrt(x);
      
      // Calcola elementi sotto la diagonale
      for (j = i + 1; j < n; ++j) {
        x = d_A[IDX(i,j,n)];
        for (k = 0; k <= i - 1; ++k) {
          x -= d_A[IDX(j,k,n)] * d_A[IDX(i,k,n)];
        }
        d_A[IDX(j,i,n)] = x * d_p[i];
      }
    }
  }
}

static void kernel_cholesky(int n,DATA_TYPE POLYBENCH_1D(p, N, n),DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){
  DATA_TYPE* d_p;
  DATA_TYPE* d_A;  // Array 1D per matrice 2D
  int* d_n;
  // Allocazione memoria GPU
  cudaMalloc(&d_p, sizeof(DATA_TYPE) * n);
  cudaMalloc(&d_A, sizeof(DATA_TYPE) * n * n);
  
  // Copia dati da host a device
  cudaMemcpy(d_p, p, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);

  // Copia matrice 2D come array 1D
  DATA_TYPE *flat_A = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      flat_A[i * n + j] = A[i][j];
    }
  }
  cudaMemcpy(d_A, flat_A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
  free(flat_A);

  /* Run kernel. */
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  kernel_cholesky<<<dimGrid, dimBlock>>>(n, d_p, d_A);

  // Sincronizzazione
  cudaDeviceSynchronize();

  // Copia risultati da device a host
  DATA_TYPE *result_A = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * n * n);
  cudaMemcpy(result_A, d_A, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(p, d_p, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);
  
  // Ricostruzione matrice 2D
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = result_A[i * n + j];
    }
  }
  free(result_A);
  /* Be clean. */
  cudaFree(d_p);
  cudaFree(d_A);
}
#endif

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(p);

  return 0;
}
