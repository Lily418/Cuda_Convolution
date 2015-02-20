
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
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

const int ITERS = 100;

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
__global__ void convolve(float* data_in, float initial)
{
    //int tx = threadIdx.x;
    //int bk = blockIdx.x;
}

/*
* Reference CPU implementation, taken from http://www.songho.ca/dsp/convolution/convolution.html
*/
bool convolve1D(std::vector<float> in, std::vector<float> out, std::vector<float> kernel)
{

    int i, j, k;

    // start convolution from out[kernelSize-1] to out[dataSize-1] (last)
    for(i = kernel.size() -1; i < in.size(); ++i)
    {
        out[i] = 0;                             // init to 0 before accumulate

        for(j = i, k = 0; k < kernel.size(); --j, ++k)
        {
            out[i] += in[j] * kernel[k];
        }
    }

    // convolution from out[0] to out[kernelSize-2]
    for(i = 0; i < kernel.size() - 1; ++i)
    {
        out[i] = 0;                             // init to 0 before sum

        for(j = i, k = 0; j >= 0; --j, ++k)
        {
            out[i] += in[j] * kernel[k];
        }
    }

    return true;
}

void splitFloats(string line, std::vector<float> floats){
    char *c_line = &line[0];
    char *endOfLine = &line[0] + line.length();
    char *end_pointer = (char*)malloc(sizeof(char));

    while(c_line < endOfLine){
        floats.push_back(strtof(c_line, &end_pointer));
        c_line = end_pointer;
    }
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

    string line;
    ifstream sample ("sample.txt");
    getline (sample,line);
    sample.close();

    // allocate host memory
    std::vector<float>  in;
    //splitFloats(line, in);
    std::vector<float>  out;

    for(int i = 0; i < 5; i++){
        in.push_back(0.0);
    }

    for(int i = 0; i < in.size(); i++){
        out.push_back(0.0);
    }

    std::vector<float> k;
    k.push_back(2);
    k.push_back(1);

    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    //convolve1D(in, out, k);

    for(int i = 0; i < 5; i++)
    {
        printf("%f, ", out[i]);
    }

    printf("\n");

    cutilCheckError(cutStopTimer(timer));
    //printf("%d \n", success);

    // allocate device memory
    //float* d_data_in;
    //cutilSafeCall(cudaMalloc((void**) &d_data_in, mem_size));

    // copy host memory to device


    // set up kernel for execution
    //printf("Run %d Kernels.\n\n", ITERS);
    //unsigned int timer = 0;
    //cutilCheckError(cutCreateTimer(&timer));
    //cutilCheckError(cutStartTimer(timer));

    // execute kernel
    //for (int j = 0; j < ITERS; j++)
    //{
    //  cutilSafeCall(cudaMemcpy(d_data_in, h_data_in,
    //  mem_size, cudaMemcpyHostToDevice));

    //   reduce<<<GRID_SIZE, BLOCK_SIZE >>>(d_data_in, 0.0);
    //   reduce<<<GRID_SIZE, BLOCK_SIZE / 2>>>(d_data_in, 0.0);
    //   reduce<<<GRID_SIZE, BLOCK_SIZE / 4>>>(d_data_in, 0.0);


    // copy result from device to host
    //   cutilSafeCall(cudaMemcpy(h_data_out, d_data_in,
    //   mem_size, cudaMemcpyDeviceToHost));

    // Finish the reduction on the host to avoid the overhead of setting up the kernal for small n
    //h_data_out[0] = host_reduce(h_data_out, 0.0, VECTOR_SIZE / 8);

    //  }

    // check if kernel execution generated and error
    //  cutilCheckMsg("Kernel execution failed");

    // wait for device to finish
    //  cudaThreadSynchronize();

    // stop and destroy timer
    //  cutilCheckError(cutStopTimer(timer));
    //  double dSeconds = cutGetTimerValue(timer)/(1000.0);
    //  double dNumOps = ITERS * size;
    //  double gflops = dNumOps/dSeconds/1.0e9;

    //Log througput
    //  printf("Throughput = %.4f GFlop/s\n", gflops);
    //  cutilCheckError(cutDeleteTimer(timer));



    // error check
    //  printf("Host reduce   : %.4f\n", host_reduce(h_data_in, 0.0, VECTOR_SIZE));
    //  printf("Device reduce : %.4f\n", h_data_out[0]);

    // clean up memory
    //  free(h_data_in);
    //  free(h_data_out);
    //  cutilSafeCall(cudaFree(d_data_in));

    // exit and clean up device status
    //  cudaThreadExit();
}
