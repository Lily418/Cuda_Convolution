
/*
 * Week 3
 * Parallel Programming
 * 2011-2012
 * University of Birmingham
 *
 * This is a first step towards implementing "parallel reduce".
 * Reducing means using an operation to aggregate the values of
 * a data type, such an array or a list.
 *
 * For example, to calculate the sum we aggregate addition:
 *     a1 + a2 + a3 + a4 ...
 * To calculate the maximum we aggregate the max operation:
 *     max (a1, max(a2, max(a3, ...
 * Note that the order in which the device map, which is parallel,
 * and the host map, which is sequential, will differ, therefore the
 * operation needs to be associative.
 * Operations such as +, * or max are associative, but function of
 * two arguments, in general, are not!
 */


#include "cutil_inline.h"

// YOU MAY WANT TO CHANGE THIS
#define BLOCK_SIZE 1024
#define GRID_SIZE 1

// DO NOT CHANGE THESE
#define VECTOR_SIZE 1024
#define ITERS 100
void datainit(float*, int);

/*
 * Function to be reduced on the device
 * Eg: sum
 */
__device__ float d_f(float x, float y) { return x + y; }

/*
 * Function to be reduced on the host
 * Eg: sum
 *
 */
float h_f(float x, float y) { return x + y; }

/*
 * Calculate the reduce by f of all elements in data_in and
 * store the result at a location of your choice in in data_out.
 *
 * The initial implementation is correct but totally sequential,
 * and it only uses one thread.
 * Improve it, to take advantage of GPU parallelism.
 * To ensure performance, identify and avoid divergences!
 *
 * THIS YOU NEED TO CHANGE!
 */
__global__ void reduce(float* data_in, float initial)
{
  int tx = threadIdx.x;
  int bk = blockIdx.x;

  int x = (bk * BLOCK_SIZE * 2) + (tx * 2);

  data_in[bk * BLOCK_SIZE + tx] = d_f(data_in[x], data_in[x + 1]);

  __syncthreads();
}

/*
 * Reference CPU implementation
 */
float host_reduce(float* data_in, float initial, int size)
{
  float rezult = initial;

  for (int k=0; k < size; k++)
      rezult = h_f(rezult, data_in[k]);

  return rezult;
}

/*
 * Main program and benchmarking
 */
int main(int argc, char** argv)
{
  int devID;
  cudaDeviceProp props;

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  // allocate host memory
  unsigned int size     = VECTOR_SIZE;              // YOU MAY WANT TO CHANGE THIS
  unsigned int mem_size = sizeof(float) * size;     // YOU MAY WANT TO CHANGE THIS
  float* h_data_in      = (float*)malloc(mem_size); // YOU MAY WANT TO CHANGE THIS
  float* h_data_out     = (float*)malloc(mem_size); // YOU MAY WANT TO CHANGE THIS

  printf("Input size : %d\n", VECTOR_SIZE);
  printf("Initial Grid size  : %d\n", GRID_SIZE);
  printf("Initial Block size : %d\n", BLOCK_SIZE);

  datainit(h_data_in, size);

  // allocate device memory
  float* d_data_in;
  cutilSafeCall(cudaMalloc((void**) &d_data_in, mem_size));

  // copy host memory to device


  // set up kernel for execution
  printf("Run %d Kernels.\n\n", ITERS);
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // execute kernel
  for (int j = 0; j < ITERS; j++)
  {
     cutilSafeCall(cudaMemcpy(d_data_in, h_data_in,
      mem_size, cudaMemcpyHostToDevice));

     reduce<<<GRID_SIZE, BLOCK_SIZE >>>(d_data_in, 0.0);
     reduce<<<GRID_SIZE, BLOCK_SIZE / 2>>>(d_data_in, 0.0);
     reduce<<<GRID_SIZE, BLOCK_SIZE / 4>>>(d_data_in, 0.0);


     // copy result from device to host
     cutilSafeCall(cudaMemcpy(h_data_out, d_data_in,
     mem_size, cudaMemcpyDeviceToHost));

     // Finish the reduction on the host to avoid the overhead of setting up the kernal for small n
     h_data_out[0] = host_reduce(h_data_out, 0.0, VECTOR_SIZE / 8);

  }

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  // wait for device to finish
  cudaThreadSynchronize();

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/(1000.0);
  double dNumOps = ITERS * size;
  double gflops = dNumOps/dSeconds/1.0e9;

  //Log througput
  printf("Throughput = %.4f GFlop/s\n", gflops);
  cutilCheckError(cutDeleteTimer(timer));



  // error check
  printf("Host reduce   : %.4f\n", host_reduce(h_data_in, 0.0, VECTOR_SIZE));
  printf("Device reduce : %.4f\n", h_data_out[0]);

  // clean up memory
  free(h_data_in);
  free(h_data_out);
  cutilSafeCall(cudaFree(d_data_in));

  // exit and clean up device status
  cudaThreadExit();
}

// Allocates a matrix with random float entries.
void datainit(float* data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand()/(float)RAND_MAX;
}
