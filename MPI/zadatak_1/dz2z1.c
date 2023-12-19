#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0
#define N 8
#define TOO_MANY_PROCESSES 1

typedef struct Result
{
    unsigned int arithmetic_count;
    unsigned int composite_count;
    unsigned int n;
    double execution_time;
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

void sequential_implementation(char **argv, Result *sequentialResult)
{
    int num = atoi(argv[1]);
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int n;

    double start_time = MPI_Wtime();
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
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    sequentialResult->arithmetic_count = arithmetic_count;
    sequentialResult->composite_count = composite_count;
    sequentialResult->n = n;
    sequentialResult->execution_time = execution_time;
}

void parallel_implementation(char **argv, Result *parallelResult)
{
    int num;
    unsigned int local_arithmetic_count = 0, global_arithmetic_count = 0;
    unsigned int local_composite_count = 0, local_master_composite_count, global_composite_count = 0;
    unsigned int n = 1;
    unsigned int i;
    unsigned int global_start = 1, local_start, local_end, chunk_size;
    unsigned int number_of_iterations;
    int communicator_size, process_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (communicator_size > N)
    {
        MPI_Abort(MPI_COMM_WORLD, TOO_MANY_PROCESSES);
    }

    if (process_rank == MASTER)
    {
        num = atoi(argv[1]);
    }

    double start_time, end_time;
    if (process_rank == MASTER)
    {
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&num, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    while (global_arithmetic_count <= num)
    {
        number_of_iterations = num + 1 - global_arithmetic_count;
        chunk_size = (number_of_iterations + communicator_size - 1) / communicator_size;
        local_start = global_start + process_rank * chunk_size;
        local_end = (local_start + chunk_size < global_start + number_of_iterations ? local_start + chunk_size : global_start + number_of_iterations);
        local_arithmetic_count = 0;
        local_composite_count = 0;
        for (i = local_start; i < local_end; i++)
        {
            unsigned int divisor_count;
            unsigned int divisor_sum;
            divisor_count_and_sum(i, &divisor_count, &divisor_sum);
            if (divisor_sum % divisor_count != 0)
                continue;
            ++local_arithmetic_count;
            if (divisor_count > 2)
                ++local_composite_count;
        }
        MPI_Allreduce(MPI_IN_PLACE, &local_arithmetic_count, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&local_composite_count, &local_master_composite_count, 1, MPI_UNSIGNED, MPI_SUM, MASTER, MPI_COMM_WORLD);
        global_arithmetic_count += local_arithmetic_count;
        global_start += number_of_iterations;
        if (process_rank == MASTER)
        {
            global_composite_count += local_master_composite_count;
            n += number_of_iterations;
        }
    }
    if (process_rank == MASTER)
    {
        end_time = MPI_Wtime();
        parallelResult->arithmetic_count = global_arithmetic_count;
        parallelResult->composite_count = global_composite_count;
        parallelResult->n = n;
        parallelResult->execution_time = end_time - start_time;
    }
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
    MPI_Init(&argc, &argv);
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    Result *sequentialResult, *parallelResult = malloc(sizeof(Result));

    if (processRank == MASTER)
    {
        /*
         * The main purpose of executing sequential implementation here is to retrieve its results so
         * that they can be compared with the results of the parallel implementation. The execution time
         * of the sequential implementation here is not relevant because it is executed inside of the MPI
         * world, so it is much slower than the actual sequential implementation executed outside of the
         * MPI world.
         */
        sequentialResult = (Result *)malloc(sizeof(Result));
        sequential_implementation(argv, sequentialResult);
        printf("Sequential implementation execution time: %fs\n", sequentialResult->execution_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    parallel_implementation(argv, parallelResult);
    if (processRank == MASTER)
    {
        printf("Parallel implementation execution time: %fs\n", parallelResult->execution_time);
        if (are_results_equal(sequentialResult, parallelResult))
            printf("Test PASSED\n");
        else
            printf("Test FAILED\n");
        free(sequentialResult);
    }

    free(parallelResult);

    MPI_Finalize();
    return 0;
}