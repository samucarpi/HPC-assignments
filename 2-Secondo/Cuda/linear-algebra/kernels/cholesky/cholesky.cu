#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "cholesky.h"

#define BLOCK_SIZE 512
#define STREAMS_SIZE 4

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

/* DCE code. Must scan the entire live-out data. Can be used also to check the correctness of the output. */
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
  #ifndef OPTIMIZED_v3
  fprintf(stderr, "\nVALORI DIAGONALI:\n");
  for (i = 0; i < n; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, 1/p[i]);
    if (i < n - 1)
      fprintf(stderr, " ");
  }
  #endif
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed, including the call and return.*/
#ifdef SEQUENTIAL
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
#endif

// Versione base 
#ifdef OPTIMIZED_v1

  // Kernel di supporto -> calcola le somme parziali A[i][k]^2
  __global__ void partial_sums(DATA_TYPE* partial, DATA_TYPE* __restrict__ A, int n, int i){
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j<=i-1){
      partial[j] = A[i*n+j]*A[i*n+j];
    }else{
      partial[j] = 0.0;
    }
  }

  __global__ void compute_p(DATA_TYPE* __restrict__ p, DATA_TYPE* partial, DATA_TYPE* __restrict__ A, int n, int i){
    int tid = threadIdx.x;
    if (tid == 0){
      DATA_TYPE sum = 0.0;
      for (int k=0; k<=i-1; ++k){
        sum += partial[k];
      }
      DATA_TYPE x = A[i*n+i]-sum;
      p[i] = 1.0/sqrt(x);
    }
  }

  __global__ void compute_column(DATA_TYPE* __restrict__ p, DATA_TYPE* __restrict__ A, int n, int i){
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j>i && j<n) {
      DATA_TYPE x = A[i*n+j];
      for (int k=0; k<=i-1; ++k){
        x -= A[j*n+k]*A[i*n+k];
      }
      A[j*n+i] = x*p[i];
    }
  }

  static void kernel_cholesky(int n, DATA_TYPE POLYBENCH_1D(p, N, n), DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){
    DATA_TYPE *d_p, *d_A, *d_partialSum;
    size_t psize = n*sizeof(DATA_TYPE);
    size_t Asize = n*n*sizeof(DATA_TYPE);
    // Allocazione memoria su device e copia dati
    cudaMalloc((void **)&d_p, psize);
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_partialSum, psize);
    cudaMemcpy(d_A, &A[0][0], Asize, cudaMemcpyHostToDevice);
    // Dimensioni griglia e blocco
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((n+BLOCK_SIZE-1)/BLOCK_SIZE);

    for (int i=0; i<n; ++i) {
      partial_sums<<<dimGrid, dimBlock>>>(d_partialSum, d_A, n, i);
      compute_p<<<1, 1>>>(d_p, d_partialSum, d_A, n, i);
      compute_column<<<dimGrid, dimBlock>>>(d_p, d_A, n, i);
    }
    cudaDeviceSynchronize();
    // Copia risultati da device a host
    cudaMemcpy(&A[0][0], d_A, Asize, cudaMemcpyDeviceToHost); 
    cudaMemcpy(p, d_p, psize, cudaMemcpyDeviceToHost);
    // Clean up memoria
    cudaFree(d_p);
    cudaFree(d_A);
    cudaFree(d_partialSum);
  }
#endif

// Versione ottimizzata con stream, shared memory e tiling
#ifdef OPTIMIZED_v2

  __global__ void compute_diagonal(DATA_TYPE* __restrict__ p, DATA_TYPE* __restrict__ A, int n, int i){
    __shared__ DATA_TYPE sharedSum[BLOCK_SIZE];
    int tid = threadIdx.x;
    
    // 1) ogni thread calcola somma parziale con stride (per i > BLOCK_SIZE)
    DATA_TYPE localSum = 0.0;
    for (int k=tid; k<i; k+=blockDim.x){
      localSum += A[i*n+k]*A[i*n+k];
    }
    sharedSum[tid] = localSum;
    __syncthreads();
    
    // 2) somma parziale parallela riducendo a 32 valori finali (warp size)
    if (tid<32){
      DATA_TYPE partialSum = 0.0;
      for (int t=tid; t<blockDim.x; t+=32){
        partialSum += sharedSum[t];
      }
      sharedSum[tid] = partialSum;
    }
    __syncthreads();
    
    // 3) somma finale dei 32 valori rimanenti seqnzialmente nel thread 0
    if (tid == 0){
      DATA_TYPE sum = 0.0;
      for (int t=0; t<32; t++){
        sum += sharedSum[t];
      }
      p[i] = 1.0/sqrt(A[i*n+i] - sum);
    }
  }

  __global__ void compute_column(DATA_TYPE* __restrict__ p, DATA_TYPE* __restrict__ A, int n, int i){
    int j = blockIdx.x*blockDim.x+threadIdx.x;  // Ogni thread calcola una colonna j
    int tid = threadIdx.x;
    __shared__ DATA_TYPE sharedPivotRow[BLOCK_SIZE];   // A[i][0..i-1]
    __shared__ DATA_TYPE sharedP;                      // p[i]
    
    // Thread 0 carica p[i] in shared memory
    if (tid == 0){
      sharedP = p[i];
    }

    // Inizializza x = A[i][j] (solo per il triangolo inferiore)
    DATA_TYPE x = 0.0;
    if (j>i && j<n){
      x = A[i*n+j];
    }
    
    // Tiling sulla riga pivot A[i][k] per sfruttare la shared memory
    for (int tile=0; tile<i; tile+=BLOCK_SIZE){
      // Carica tile della riga pivot: tutti i thread collaborano
      int k_curr = tile+tid;
      if (k_curr<i){
        sharedPivotRow[tid] = A[i*n+k_curr];
      }else{
        sharedPivotRow[tid] = 0.0;  // Padding per evitare accessi fuori limite
      }
      __syncthreads();  // Attende il caricamento completo della riga pivot in shared memory
      
      // Ogni thread calcola: x -= Σ A[j][k] * A[i][k] per k nel tile corrente
      if (j>i && j<n){
        int pivotRowSize = min(BLOCK_SIZE,i-tile);  // Numero di elementi validi nel tile corrente
        for (int k=0; k<pivotRowSize; ++k){
          x -= A[j*n+(tile+k)]*sharedPivotRow[k];  // A[i][k] da shared memory A[j][k] da global memory
        }
      }
      __syncthreads();  // Attende che tutti i thread abbiano finito di usare la riga pivot corrente
    }
    
    // Scrivi risultato finale: A[j][i] = x * p[i]
    if (j>i && j<n){
      A[j*n+i] = x*sharedP;
    }
  }

  static void kernel_cholesky(int n, DATA_TYPE POLYBENCH_1D(p, N, n), DATA_TYPE POLYBENCH_2D(A, N, N, n, n)){
    DATA_TYPE *d_p, *d_A;
    size_t psize = n*sizeof(DATA_TYPE);
    size_t Asize = n*n*sizeof(DATA_TYPE);
    // Allocazione memoria su device e copia dati
    cudaMalloc((void **)&d_p, psize);
    cudaMalloc((void **)&d_A, Asize);
    cudaMemcpy(d_A, &A[0][0], Asize, cudaMemcpyHostToDevice);
    // Dimensioni griglia e blocco
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((n+BLOCK_SIZE-1)/BLOCK_SIZE);
    for (int i=0; i<n; ++i) {
      // dimGrid = 1 per avere la shared memory condivisa per tutti i thread del blocco. Computa anche elementi > BLOCK_SIZE grazie al loop stride.
      // La dimensione della shared memory è statica e definita da BLOCK_SIZE all'interno dei kernel.
      compute_diagonal<<<1, dimBlock>>>(d_p, d_A, n, i); 
      compute_column<<<dimGrid, dimBlock>>>(d_p, d_A, n, i);
    }
    // Sincronizza tutti gli stream per garantire il completamento di tutte le operazioni
    cudaDeviceSynchronize();
    // Copia risultati da device a host
    cudaMemcpy(&A[0][0], d_A, Asize, cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_p, psize, cudaMemcpyDeviceToHost);
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
  #ifdef OPTIMIZED_v3
    kernel_cholesky(n, POLYBENCH_ARRAY(A));
  #else
    kernel_cholesky(n, POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(A));
  #endif
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

/*
CONTROLLO SUI RISULTATI (custom_matrix.txt):

  MATRICE RISULTANTE:
  4.00  2.00  1.00  1.00 
  1.00  5.00  2.00  1.00 
  0.50  0.75  6.00  2.00 
  0.50  0.25  0.69  7.00 

  VALORI DIAGONALI:
  2.00  2.00  2.28  2.49 
*/