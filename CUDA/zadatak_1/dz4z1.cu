#include <stdio.h>
#include <stdlib.h>

#define NUMBER_OF_THREADS_PER_BLOCK 256

__device__ void divisor_count_and_sum(unsigned int n, unsigned int *pcount, unsigned int *psum)
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

__global__ void findArithmeticNumbersKernel(unsigned int *arithmetic_count_gpu, unsigned int *composite_count_gpu, int start, int number_of_iterations)
{
    if (blockIdx.x * blockDim.x + threadIdx.x < number_of_iterations)
    {
        unsigned int divisor_count;
        unsigned int divisor_sum;
        unsigned int myNumber = start + (blockIdx.x * blockDim.x + threadIdx.x);
        divisor_count_and_sum(myNumber, &divisor_count, &divisor_sum);
        if (divisor_sum % divisor_count == 0)
        {
            atomicAdd(arithmetic_count_gpu, 1);
            if (divisor_count > 2)
            {
                atomicAdd(composite_count_gpu, 1);
            }
        }
    }
}

int main(int argc, char **argv)
{
    int num = atoi(argv[1]);
    unsigned int arithmetic_count_cpu = 0, *arithmetic_count_gpu;
    unsigned int composite_count_cpu = 0, *composite_count_gpu;
    unsigned int n = 1;
    unsigned int start = 1;
    unsigned int number_of_iterations = 1;

    cudaMalloc(&arithmetic_count_gpu, sizeof(unsigned int));
    cudaMalloc(&composite_count_gpu, sizeof(unsigned int));

    cudaMemset(arithmetic_count_gpu, 0, sizeof(unsigned int));
    cudaMemset(composite_count_gpu, 0, sizeof(unsigned int));

    while (arithmetic_count_cpu <= num)
    {
        number_of_iterations = num + 1 - arithmetic_count_cpu;
        n += number_of_iterations;

        printf("number_of_iterations: %d\n", number_of_iterations);

        dim3 gridDimension((number_of_iterations + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK);
        dim3 blockDimension(NUMBER_OF_THREADS_PER_BLOCK);

        findArithmeticNumbersKernel<<<gridDimension, blockDimension>>>(arithmetic_count_gpu, composite_count_gpu, start, number_of_iterations);

        cudaMemcpy(&arithmetic_count_cpu, arithmetic_count_gpu, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&composite_count_cpu, composite_count_gpu, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        start += number_of_iterations;
    }

    printf("\n%uth arithmetic number is %u\n", arithmetic_count_cpu, n);
    printf("Number of composite arithmetic numbers <= %u: %u\n", n, composite_count_cpu);

    return 0;
}