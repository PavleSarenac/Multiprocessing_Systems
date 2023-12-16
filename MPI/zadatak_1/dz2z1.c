#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0
#define N 8
#define TOO_MANY_PROCESSES 1

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

int main(int argc, char **argv)
{
    int num;
    unsigned int local_arithmetic_count = 0, global_arithmetic_count = 0;
    unsigned int local_composite_count = 0, local_master_composite_count, global_composite_count = 0;
    unsigned int n = 1;
    unsigned int i;
    unsigned int global_start = 1, local_start, local_end, chunk_size;
    unsigned int number_of_iterations;
    int communicator_size, process_rank;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (communicator_size > N)
    {
        MPI_Abort(MPI_COMM_WORLD, TOO_MANY_PROCESSES);
    }

    if (process_rank == MASTER)
    {
        num = atoi(argv[1]);
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
        printf("\n%uth arithmetic number is %u\n", global_arithmetic_count, n);
        printf("Number of composite arithmetic numbers <= %u: %u\n", n, global_composite_count);
        printf("Execution time: %fs\n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}