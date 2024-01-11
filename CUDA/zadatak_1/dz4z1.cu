#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUMBER_OF_THREADS_PER_BLOCK 256

typedef struct Result
{
    int arithmetic_count;
    int composite_count;
    int n;
    float execution_time;
} Result;

__device__ __host__ void divisor_count_and_sum(unsigned int n, unsigned int *pcount, unsigned int *psum)
{
    unsigned int divisor_count = 1;
    unsigned int divisor_sum = 1;
    unsigned int power = 2;
    for (; (n & 1) == 0; power <<= 1, n >>= 1)
    {
        ++divisor_count;
        divisor_sum += power;
    }
    for (unsigned int p = 3; p * p <= n; p += 2)
    {
        unsigned int count = 1, sum = 1;
        for (power = p; n % p == 0; power *= p, n /= p)
        {
            ++count;
            sum += power;
        }
        divisor_count *= count;
        divisor_sum *= sum;
    }
    if (n > 1)
    {
        divisor_count *= 2;
        divisor_sum *= n + 1;
    }
    *pcount = divisor_count;
    *psum = divisor_sum;
}

__global__ void findArithmeticNumbersKernel(unsigned int *arithmetic_count_array_gpu, unsigned int *composite_count_array_gpu, int start, int number_of_iterations)
{
    __shared__ unsigned int arithmeticNumbersCounters[NUMBER_OF_THREADS_PER_BLOCK];
    __shared__ unsigned int compositeNumbersCounters[NUMBER_OF_THREADS_PER_BLOCK];

    arithmeticNumbersCounters[threadIdx.x] = 0;
    compositeNumbersCounters[threadIdx.x] = 0;

    if (blockIdx.x * blockDim.x + threadIdx.x < number_of_iterations)
    {
        unsigned int divisor_count;
        unsigned int divisor_sum;
        unsigned int myNumber = start + (blockIdx.x * blockDim.x + threadIdx.x);
        divisor_count_and_sum(myNumber, &divisor_count, &divisor_sum);
        if (divisor_sum % divisor_count == 0)
        {
            arithmeticNumbersCounters[threadIdx.x] = 1;
            if (divisor_count > 2)
            {
                compositeNumbersCounters[threadIdx.x] = 1;
            }
        }
    }

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (threadIdx.x < i)
        {
            arithmeticNumbersCounters[threadIdx.x] += arithmeticNumbersCounters[threadIdx.x + i];
            compositeNumbersCounters[threadIdx.x] += compositeNumbersCounters[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        arithmetic_count_array_gpu[blockIdx.x] = arithmeticNumbersCounters[0];
        composite_count_array_gpu[blockIdx.x] = compositeNumbersCounters[0];
    }
}

Result *arithmeticNumbersCPU(char **argv)
{
    Result *result = (Result *)malloc(sizeof(Result));

    int num = atoi(argv[1]);
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int n;

    struct timespec start_time, end_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (n = 1; arithmetic_count <= num; ++n)
    {
        unsigned int divisor_count;
        unsigned int divisor_sum;
        divisor_count_and_sum(n, &divisor_count, &divisor_sum);
        if (divisor_sum % divisor_count != 0)
            continue;
        ++arithmetic_count;
        if (divisor_count > 2)
            ++composite_count;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    result->arithmetic_count = arithmetic_count;
    result->composite_count = composite_count;
    result->n = n;
    result->execution_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    return result;
}

Result *arithmeticNumbersGPU(char **argv)
{
    Result *result = (Result *)malloc(sizeof(Result));

    int num = atoi(argv[1]);
    unsigned int *arithmetic_count_array_cpu, *arithmetic_count_array_gpu;
    unsigned int *composite_count_array_cpu, *composite_count_array_gpu;
    unsigned int arithmetic_count = 0, composite_count = 0;
    unsigned int n = 1;
    unsigned int start = 1;
    unsigned int number_of_iterations = 1;
    int counterArraysSize = ((num + 1 + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK) *
                            sizeof(unsigned int);

    // Dummy call - purpose is to set up CUDA environment here so that initialization overhead isn't included in profiling
    // statistics of actual useful CUDA API calls.
    cudaDeviceSynchronize();

    cudaEvent_t start_time = cudaEvent_t();
    cudaEvent_t end_time = cudaEvent_t();
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    cudaEventRecord(start_time, 0);

    arithmetic_count_array_cpu = (unsigned int *)malloc(counterArraysSize);
    composite_count_array_cpu = (unsigned int *)malloc(counterArraysSize);

    cudaMalloc(&arithmetic_count_array_gpu, counterArraysSize);
    cudaMalloc(&composite_count_array_gpu, counterArraysSize);

    while (arithmetic_count <= num)
    {
        number_of_iterations = num + 1 - arithmetic_count;
        n += number_of_iterations;

        unsigned int numberOfBlocks = (number_of_iterations + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK;

        dim3 gridDimension(numberOfBlocks);
        dim3 blockDimension(NUMBER_OF_THREADS_PER_BLOCK);

        findArithmeticNumbersKernel<<<gridDimension, blockDimension>>>(arithmetic_count_array_gpu, composite_count_array_gpu, start, number_of_iterations);

        cudaMemcpy(arithmetic_count_array_cpu, arithmetic_count_array_gpu, numberOfBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(composite_count_array_cpu, composite_count_array_gpu, numberOfBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numberOfBlocks; i++)
        {
            arithmetic_count += arithmetic_count_array_cpu[i];
            composite_count += composite_count_array_cpu[i];
        }

        start += number_of_iterations;
    }

    cudaFree(arithmetic_count_array_gpu);
    cudaFree(composite_count_array_gpu);

    free(arithmetic_count_array_cpu);
    free(composite_count_array_cpu);

    cudaEventRecord(end_time, 0);
    cudaEventSynchronize(end_time);
    float execution_time;
    cudaEventElapsedTime(&execution_time, start_time, end_time);

    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);

    result->arithmetic_count = arithmetic_count;
    result->composite_count = composite_count;
    result->n = n;
    result->execution_time = execution_time / 1000;

    return result;
}

int are_results_equal(Result *sequential_result, Result *parallel_result)
{
    if (sequential_result->arithmetic_count == parallel_result->arithmetic_count &&
        sequential_result->composite_count == parallel_result->composite_count &&
        sequential_result->n == parallel_result->n)
        return 1;
    return 0;
}

int main(int argc, char **argv)
{
    Result *sequential_result, *parallel_result;

    sequential_result = arithmeticNumbersCPU(argv);
    parallel_result = arithmeticNumbersGPU(argv);

    printf("Sequential implementation execution time: %fs\n", sequential_result->execution_time);
    printf("Parallel implementation execution time: %fs\n", parallel_result->execution_time);
    if (are_results_equal(sequential_result, parallel_result))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    free(sequential_result);
    free(parallel_result);

    return 0;
}