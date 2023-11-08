#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct Result
{
    unsigned int arithmetic_count;
    unsigned int composite_count;
    unsigned int n;
    double execution_time;
    unsigned int number_of_threads;
} Result;

void divisor_count_and_sum(unsigned int n, unsigned int *pcount,
                           unsigned int *psum)
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

Result *sequential_implementation(char **argv)
{
    int num = atoi(argv[1]);
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int n;

    double start_time = omp_get_wtime();
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
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    Result *result = malloc(sizeof(Result));
    if (result)
    {
        result->arithmetic_count = arithmetic_count;
        result->composite_count = composite_count;
        result->n = n;
        result->execution_time = execution_time;
        result->number_of_threads = 1;
    }
    return result;
}

Result *parallel_implementation(char **argv, int number_of_threads)
{
    int num = atoi(argv[1]);
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int n = 1;
    unsigned int i;
    unsigned int start = 1;
    unsigned int number_of_iterations;

    double start_time = omp_get_wtime();
    while (arithmetic_count <= num)
    {
        number_of_iterations = num + 1 - arithmetic_count;
        n += number_of_iterations;
#pragma omp parallel for default(none) private(i) shared(start, number_of_iterations)             \
    reduction(+ : arithmetic_count) reduction(+ : composite_count) num_threads(number_of_threads) \
    schedule(guided)
        for (i = start; i < start + number_of_iterations; i++)
        {
            unsigned int divisor_count;
            unsigned int divisor_sum;
            divisor_count_and_sum(i, &divisor_count, &divisor_sum);
            if (divisor_sum % divisor_count != 0)
                continue;
            ++arithmetic_count;
            if (divisor_count > 2)
                ++composite_count;
        }
        start += number_of_iterations;
    }
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    Result *result = malloc(sizeof(Result));
    if (result)
    {
        result->arithmetic_count = arithmetic_count;
        result->composite_count = composite_count;
        result->n = n;
        result->execution_time = execution_time;
        result->number_of_threads = number_of_threads;
    }
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
    Result *sequential_result = sequential_implementation(argv);
    printf("Sequential implementation execution time: %fs\n", sequential_result->execution_time);

    Result *parallel_result_one_thread = parallel_implementation(argv, 1);
    printf("Parallel implementation (one thread) execution time: %fs\n", parallel_result_one_thread->execution_time);
    if (are_results_equal(sequential_result, parallel_result_one_thread))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    Result *parallel_result_two_threads = parallel_implementation(argv, 2);
    printf("Parallel implementation (two threads) execution time: %fs\n", parallel_result_two_threads->execution_time);
    if (are_results_equal(sequential_result, parallel_result_two_threads))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    Result *parallel_result_four_threads = parallel_implementation(argv, 4);
    printf("Parallel implementation (four threads) execution time: %fs\n", parallel_result_four_threads->execution_time);
    if (are_results_equal(sequential_result, parallel_result_four_threads))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    Result *parallel_result_eight_threads = parallel_implementation(argv, 8);
    printf("Parallel implementation (eight threads) execution time: %fs\n", parallel_result_eight_threads->execution_time);
    if (are_results_equal(sequential_result, parallel_result_eight_threads))
        printf("Test PASSED\n\n");
    else
        printf("Test FAILED\n\n");

    free(sequential_result);
    free(parallel_result_one_thread);
    free(parallel_result_two_threads);
    free(parallel_result_four_threads);
    free(parallel_result_eight_threads);
    return 0;
}