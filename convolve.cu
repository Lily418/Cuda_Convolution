
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

#define BLOCK_SIZE 5
#define GRID_SIZE 1
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
__global__ void convolve(float* data_in, float* data_out, float* kernel, int kernelSize)
{
    int tx = threadIdx.x;
    int bk = blockIdx.x;
    int pos = (bk * BLOCK_SIZE) + tx;
    data_out[pos] = 0;

    for(int i = 0; i < kernelSize; i++){
        if(pos - i >= 0) {
            data_out[pos] += kernel[i] * data_in[pos - i];
        }
    }

}

/*
* Reference CPU implementation, taken from http://www.songho.ca/dsp/convolution/convolution.html
*/
std::vector<float> convolve1D(std::vector<float> in, std::vector<float> kernel)
{

    std::vector<float> out;
    for(int i = 0; i < in.size(); i++){
        out.push_back(0.0);
    }

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

    return out;
}

std::vector<float> splitFloats(string line){
    std::vector<float> floats;
    char *c_line = &line[0];
    char *endOfLine = &line[0] + line.length();
    char *end_pointer = (char*)malloc(sizeof(char));

    while(c_line < endOfLine){
        floats.push_back(strtof(c_line, &end_pointer));
        c_line = end_pointer;
    }

    return floats;
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

    string sample_line;
    string kernel_line;
    ifstream sample ("sample_small.txt");
    ifstream kernel ("kernel.txt");
    getline (sample, sample_line);
    getline (kernel, kernel_line);

    sample.close();
    kernel.close();

    // allocate host memory
    std::vector<float> in = splitFloats(sample_line);
    std::vector<float> k   = splitFloats(kernel_line);
    std::vector<float> out;

    float* d_data_in;
    cutilSafeCall(cudaMalloc((void**) &d_data_in, sizeof(float) * in.size()));
    cutilSafeCall(cudaMemcpy(d_data_in, &in[0], sizeof(float) * in.size(), cudaMemcpyHostToDevice));

    float* d_data_out;

    for(int i = 0; i < in.size(); i++){
        out.push_back(0.0);
    }
    cutilSafeCall(cudaMalloc((void**) &d_data_out, sizeof(float) * out.size()));
    cutilSafeCall(cudaMemcpy(d_data_out, &out[0], sizeof(float) * out.size(), cudaMemcpyHostToDevice));

    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));


    cutilCheckError(cutStartTimer(timer));

    if(argc != 2){
        printf("Please pass command line argument: 0 for CPU, 1 for GPU, 2 for GPU Optimised \n");
        return 1;
    }

    for (int i = 0; i < ITERS; i++)
    {
        switch(atoi(argv[1])){
            case 0:
            out = convolve1D(in, k);
            break;

            case 1:
            convolve<<<GRID_SIZE, BLOCK_SIZE >>>(d_data_in, d_data_out, &k[0], k.size());
            break;

            case 2:
            printf("To implement GPU-Opt");
            break;

            default:
            printf("Please pass command line argument: 0 for CPU, 1 for GPU, 2 for GPU Optimised");
            break;
        }
    }
    cutilCheckError(cutStopTimer(timer));

    ofstream convolution;
    convolution.open("convolution.txt", ios::trunc);

    if(atoi(argv[1]) == 1 || atoi(argv[1]) == 2){
    cutilSafeCall(cudaMemcpy(&out[0], d_data_out, sizeof(float) * out.size() , cudaMemcpyDeviceToHost));
    }


    for(int i = 0; i < out.size(); i++)
    {
        convolution << out[i] << " ";
    }

    convolution << std::endl;
    convolution.close();


    double dSeconds = cutGetTimerValue(timer)/(1000.0);
    double dNumOps = ITERS * in.size() * k.size();
    double gflops = dNumOps/dSeconds/1.0e9;

    printf("Throughput = %.4f GFlop/s\n", gflops);



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
