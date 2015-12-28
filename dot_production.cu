#include <stdio.h>
#include <stdlib.h>
#define min(a,b) (a<b?a:b)

#define threadsPerBlock  256
#define N  33 * 1024
#define blocksPerGrid  min(32, (N+threadsPerBlock-1)/threadsPerBlock)

__global__ void dot(float *a, float *b, float *c) {

    //calculate thread id combining the block and thread indices to get global offset into the input arrays
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //shared memory for each block, which means each block has a copy of the memory.
    //and the index of the cache is just the thread index in each block
    __shared__ float cache[threadsPerBlock];
    int cache_index = threadIdx.x;

    int i;
    float tmp = 0;
    //each thread multiplies a pair of corresponding entries, and then every thread moves on to its next pare.
    //the threads increment their indices by the total number of threads to easure we don't miss any elements and don't multiply a pair twice
    while (tid < N) {
        tmp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    //in each block, store the sum of pairs from each thread
    cache[cache_index] = tmp;
    //sync threads in the block before we sum all the values resulted from each thread.
    __syncthreads();

    //for reductions, threads Per block must be power of 2 because of the following reduction
    i = blockDim.x/2;
    while (i!=0) {
        if(cache_index < i) {
            cache[cache_index] += cache[cache_index+i];
            //__syncthreads();   THIS IS NOT ALLOWED and GPU will not work!!!!
        }
        //sync threads  after each iteration of reduction
        //notice that the "__syncthreads" cannot be placed in the above "if" block
        //because cuda architecture guarantees that no thread will advance to an instruction beyond the __syncthreads() until every
        //thread in the block has executed the "syncthreads", however, if the "__syncthreads" is placed into a divergent branch,
        //some threads block will never go to the branch and hardware will simply continue to wait for these threads, forever.
         __syncthreads();
         i/=2;
    }

    //Use one thread in each block to write the results of each block to the global memory
    //here "c" gather each block's sum results, since there is not many blocks, we don't leverage GPU to complete the final results
    //and use CPU to compute this part
    if (cache_index == 0 ) {
        c[blockIdx.x] = cache[0];
    }

}

int main(void) {
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    int i;
    float sum;

    //allocate memory for array a, b and partial_c on CPU side
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

    //initialize a and b in CPU
    for (i=0;i<N; i++){
        a[i] = i;
        b[i] = i*2;
    }

    //malloc memory on GPU for array a, array b and partial results for each block
    cudaMalloc((void **) &dev_a, N * sizeof(float) );
    cudaMalloc((void **) &dev_b, N * sizeof(float) );
    cudaMalloc((void **) &dev_partial_c, blocksPerGrid * sizeof(float) );

    //copy memory from host to device
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_partial_c, partial_c,  blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice);

    //call the kernel
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    for (i = 0; i< blocksPerGrid; i++) {
        sum += partial_c[i];
    }

    //verify whether the result is correct
    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    float expect_value = 2 * sum_squares((float)(N-1));

    printf("does the gpu value %.6g = %.6g\n", sum, expect_value);

    //free memory on GPU side
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    //free memory on CPU side
    free(a);
    free(b);
    free(partial_c);

    return 0;
}
