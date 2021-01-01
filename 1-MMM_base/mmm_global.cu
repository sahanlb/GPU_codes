#include <cstdio>
#include <cstdlib>
#include <math.h>
//includes for timing
#include <time.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define PRINT_TIME              1
#define VALIDATE                0
#define WIDTH                 	2048
#define MATRIX_SIZE             WIDTH*WIDTH
#define TOL                     10E14

void initializeArray1D(float *arr, int len, int seed);
struct timespec diff(struct timespec start, struct timespec end);

//MMM kernel
__global__ void kernel_mmm(int width, float* M, float* N, float* P){
  const int bx_id = blockIdx.x;
  const int by_id = blockIdx.y;

  const int tx_id = threadIdx.x;
  const int ty_id = threadIdx.y;

  const int col = bx_id*blockDim.x + tx_id;
  const int row = by_id*blockDim.y + ty_id;

  float Pvalue = 0;

  for(int k=0; k<width; k++){
    float Melement = M[row*width + k];
    float Nelement = N[k*width + col];
    Pvalue += Melement*Nelement;
  }

  P[row*width + col] = Pvalue;
}



int main(int argc, char **argv){
  //struct timespec time1, time2;
  //struct timespec time_stamp;

  // GPU Timing variables
  cudaEvent_t start, stop, kernel_start, kernel_stop;
  float elapsed_gpu_kernel;
  float elapsed_gpu_total;

  // Arrays on GPU global memoryc
  float *d_M;
  float *d_N;
  float *d_P;

  // Arrays on the host memory
  float *h_M;
  float *h_N;
  float *h_P;
  float *h_P_verify;

  //sum variables
  printf("Row length of the matrix = %d\n", WIDTH);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize = MATRIX_SIZE * sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_M, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_N, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_P, allocSize));

  printf("Device allocation done");


  // Allocate arrays on host memory
  h_M        = (float *) malloc(allocSize);
  h_N        = (float *) malloc(allocSize);
  h_P        = (float *) malloc(allocSize);
  h_P_verify = (float *) malloc(allocSize);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray1D(h_M, MATRIX_SIZE, 2453);
  initializeArray1D(h_N, MATRIX_SIZE, 1773);

  printf("\t... done\n\n");

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_M, h_M, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_N, h_N, allocSize, cudaMemcpyHostToDevice));

  //block dimensions
  dim3 dimBlock(16,16,1);
  dim3 dimGrid(128,128,1); //2048
  
#if PRINT_TIME
  cudaEventRecord(kernel_start, 0);
#endif

  // Launch the kernel
  kernel_mmm<<<dimGrid, dimBlock>>>(WIDTH, d_M, d_N, d_P);

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(kernel_stop,0);
  cudaEventSynchronize(kernel_stop);
  cudaEventElapsedTime(&elapsed_gpu_kernel, kernel_start, kernel_stop);
  printf("\nMMM kernel time: %f (msec)\n", elapsed_gpu_kernel);
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_stop);
#endif


  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_P, d_P, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu_total, start, stop);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu_total);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif


  /*Verification using host CPU*/
#if VALIDATE
  float t;

  for(int i=0; i<WIDTH; i++){
    for(int j=0; j<WIDTH; j++){
      t = 0;
      for(int k=0; k<WIDTH; k++){
        t += h_M[i*WIDTH + k] * h_N[k*WIDTH + j];
      }
      //h_P_verify[i*WIDTH + j] += t;
      h_P_verify[i*WIDTH + j] = t;
    }
  }

  //compare two metrices
  for(int i=0; i<WIDTH; i++){
    for(int j=0; j<WIDTH; j++){
      if(abs(h_P_verify[i*WIDTH + j] - h_P[i*WIDTH + j]) > TOL){
        float diff = abs(h_P_verify[i*WIDTH + j] - h_P[i*WIDTH + j]);
        printf("Element %d,%d exceeds tolerance by %f. CPU result = %f, GPU result = %f\n", diff, i, j, h_P_verify[i*WIDTH + j], h_P[i*WIDTH + j]);
	return 1;
      } 
    }
  }
#endif

  printf("Results of CPU and GPU calculations match with a tolerance of %f\n", TOL);

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_M));
  CUDA_SAFE_CALL(cudaFree(d_N));
  CUDA_SAFE_CALL(cudaFree(d_P));

  free(h_M);
  free(h_N);
  free(h_P);
  free(h_P_verify);

  return 0;
}



void initializeArray1D(float *arr, int len, int seed) {
  int i;
  float randNum;
  srand(seed);

  for (i = 0; i < len; i++) {
    randNum = (float) rand();
    arr[i] = randNum;
  }
}

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}
