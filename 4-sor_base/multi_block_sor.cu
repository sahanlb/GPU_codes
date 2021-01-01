#include <cstdio>
#include <cstdlib>
#include <math.h>
//#include "cuPrintf.cu"
//#include "cuPrintf.cuh"
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
#define ROW_LEN                 2048
#define MATRIX_SIZE             ROW_LEN*ROW_LEN
#define OMEGA                   1.0

void initializeArray1D(float *arr, int len, int seed);
struct timespec diff(struct timespec start, struct timespec end);

//SOR kernel
__global__ void kernel_sor(int arrLen, float* x, float* y){
  const int bx_id = blockIdx.x;
  const int by_id = blockIdx.y;

  const int tx_id = threadIdx.x;
  const int ty_id = threadIdx.y;

  const int col = bx_id*blockDim.x + tx_id;
  const int row = by_id*blockDim.y + ty_id;


  float change;

  if((row != 0) && (row != arrLen-1) && (col != 0) && (col != arrLen-1)){
    change = x[row*arrLen + col] - 0.25 * (x[(row-1)*arrLen + col] + x[(row+1)*arrLen + col] + x[row*arrLen + col + 1] + x[row*arrLen + col - 1]);
    y[row*arrLen + col] = x[row*arrLen + col] - (change * OMEGA);
  }
  else{
    y[row*arrLen + col] = x[row*arrLen + col];
  }

}



int main(int argc, char **argv){
  struct timespec time1, time2;
  struct timespec time_stamp;

  int arrLen = 0;

  // GPU Timing variables
  cudaEvent_t start, stop, kernel_start, kernel_stop;
  float elapsed_gpu, elapsed_gpu_kernel;

  // Arrays on GPU global memoryc
  float *d_x;
  float *d_y;

  // Arrays on the host memory
  float *h_x;
  float *h_y;
  float *h_result;

  //sum variables
  float gpu_sum, cpu_sum;
  
  float change;

  if (argc > 1) {
    arrLen  = atoi(argv[1]);
  }
  else {
    arrLen = ROW_LEN;
  }

  printf("Row length of the matrix = %d\n", arrLen);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate GPU memory
  size_t allocSize = arrLen * arrLen * sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, allocSize));

  printf("Device allocation done");


  // Allocate arrays on host memory
  h_x                        = (float *) malloc(allocSize);
  h_y                        = (float *) malloc(allocSize);
  h_result                   = (float *) malloc(allocSize);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray1D(h_x, (arrLen*arrLen), 2453);
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
  CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));

  //cudaPrintfInit();

  //block dimensions
  dim3 dimBlock(16,16,1);
  //dim3 dimGrid(64,64,1); //1024
  dim3 dimGrid(128,128,1); //2048
  // Launch the kernel
  cudaEventRecord(kernel_start, 0);
  for(int iter=0; iter<1000; iter++){
     kernel_sor<<<dimGrid, dimBlock>>>(arrLen, d_x, d_y);
     cudaDeviceSynchronize();
     kernel_sor<<<dimGrid, dimBlock>>>(arrLen, d_y, d_x);
     cudaDeviceSynchronize();
  }

  // Print kernel time
#if PRINT_TIME
  cudaEventRecord(kernel_stop,0);
  cudaEventSynchronize(kernel_stop);
  cudaEventElapsedTime(&elapsed_gpu_kernel, kernel_start, kernel_stop);
  printf("\nSOR kernel time: %f (msec)\n", elapsed_gpu_kernel);
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_stop);
#endif

  //cudaPrintfDisplay(stdout, true);
  //cudaPrintfEnd();

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_result, d_x, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("\nTotal GPU time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif


  /*Calculate sum*/
  gpu_sum = 0.0;

  for(int i=0; i<arrLen; i++){
    for(int j=0; j<arrLen; j++){
      gpu_sum += h_result[i*arrLen + j];
    }
  }

  //print sum
  printf("Sum of GPU calculated array elements: %f\n", gpu_sum);


  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

  //Calculate reference result using the CPU
  /*for(int iter=0; iter<2000; iter++){
    for(int ii=0; ii<arrLen; ii+=16){
      for(int jj=0; jj<arrLen; jj+=16){
        for(int i=ii; i<ii+16; i++){
          for(int j=jj; j<jj+16; j++){
            if((i==0) || (i==arrLen-1) || (j==0) || (j==arrLen-1)){
              continue;
            }
	    else{
              change = h_x[i*arrLen + j] - 0.25 * (h_x[(i-1)*arrLen + j] + 
                                                    h_x[(i+1)*arrLen + j] + 
                                                    h_x[i*arrLen + j - 1] + 
                                                    h_x[i*arrLen + j + 1]);
	      h_x[i*arrLen + j] -= change * OMEGA;
            }
          }
        }
      }
    }
  }*/

#if VALIDATE
  for(int iter=0; iter<1000; iter++){
    // hx -> hy
    for(int i=0; i<arrLen; i++){
      for(int j=0; j<arrLen; j++){
        if((i==0) || (i==arrLen-1) || (j==0) || (j==arrLen-1)){
          h_y[i*arrLen + j] = h_x[i*arrLen + j];
        }
        else{
          change = h_x[i*arrLen + j] - 0.25 * (h_x[(i-1)*arrLen + j] + 
                                                h_x[(i+1)*arrLen + j] + 
                                                h_x[i*arrLen + j - 1] + 
                                                h_x[i*arrLen + j + 1]);
          
          h_y[i*arrLen + j] = h_x[i*arrLen + j] - (OMEGA * change);
        }
      }
    }

    // hy -> hx
    for(int i=0; i<arrLen; i++){
      for(int j=0; j<arrLen; j++){
        if((i==0) || (i==arrLen-1) || (j==0) || (j==arrLen-1)){
          h_x[i*arrLen + j] = h_y[i*arrLen + j];
        }
        else{
          change = h_y[i*arrLen + j] - 0.25 * (h_y[(i-1)*arrLen + j] + 
                                                h_y[(i+1)*arrLen + j] + 
                                                h_y[i*arrLen + j - 1] + 
                                                h_y[i*arrLen + j + 1]);
          
          h_x[i*arrLen + j] = h_y[i*arrLen + j] - (OMEGA * change);
        }
      }
    }
  }
#endif

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  time_stamp = diff(time1,time2);

  printf("CPU run time: %ld(ns)\n", (long int)((double)(1000000000 * time_stamp.tv_sec + time_stamp.tv_nsec)));

  //Calculate sum
  cpu_sum = 0.0;

  for(int i=0; i<arrLen; i++){
    for(int j=0; j<arrLen; j++){
      cpu_sum += h_x[i*arrLen + j];
    }
  }

  //print sum
  printf("Sum of CPU calculated array elements: %f\n", cpu_sum);



  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));

  free(h_x);
  free(h_y);
  free(h_result);

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
