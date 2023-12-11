#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <omp.h>

#define SOFTENING 1e-9f
#define ACCURACY 0.01f

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

typedef struct Result
{
    double execution_time;
    int number_of_threads;
    Body *all_bodies;
} Result;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void bodyForce_sequential(Body *p, float dt, int n)
{
    for (int i = 0; i < n; i++)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

void saveToCSV(Body *p, int n, int iter, const char *folder)
{
    char filename[50];
    sprintf(filename, "%s/iteration_%d.csv", folder, iter);
    FILE *file = fopen(filename, "w");

    fprintf(file, "x,y,z,vx,vy,vz\n");
    for (int i = 0; i < n; i++)
    {
        fprintf(file, "%f,%f,%f,%f,%f,%f\n", p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
    }

    fclose(file);
}

Result *sequential_implementation(char **argv, float *initial_buf)
{
    int nBodies = atoi(argv[1]);
    int nIters = atoi(argv[2]);
    const char *folder = argv[3];

    const float dt = 0.01f;

    mkdir(folder, 0700);

    int bytes = nBodies * sizeof(Body);
    float *fakeBuf = (float *)malloc(bytes);
    float *buf = initial_buf;
    Body *p = (Body *)buf;

    double start_time = omp_get_wtime();
    randomizeBodies(fakeBuf, 6 * nBodies);

    for (int iter = 0; iter < nIters; iter++)
    {
        bodyForce_sequential(p, dt, nBodies);

        saveToCSV(p, nBodies, iter, folder);

        for (int i = 0; i < nBodies; i++)
        {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    }
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    free(fakeBuf);

    Result *result = malloc(sizeof(Result));
    if (result)
    {
        result->execution_time = execution_time;
        result->number_of_threads = 1;
        result->all_bodies = p;
    }
    return result;
}

void bodyForce_parallel(Body *p, float dt, int n, int number_of_threads)
{
#pragma omp parallel for default(none) shared(p, dt, n) num_threads(number_of_threads)
    for (int i = 0; i < n; i++)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;

            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

Result *parallel_implementation(char **argv, int number_of_threads, float *initial_buf)
{
    int nBodies = atoi(argv[1]);
    int nIters = atoi(argv[2]);
    const char *folder = argv[3];

    const float dt = 0.01f;

    mkdir(folder, 0700);

    int bytes = nBodies * sizeof(Body);
    float *fakeBuf = (float *)malloc(bytes);
    float *buf = initial_buf;
    Body *p = (Body *)buf;

    int iter = 0;
    FILE *file;

    double start_time = omp_get_wtime();
    randomizeBodies(fakeBuf, 6 * nBodies);
    for (iter = 0; iter < nIters; iter++)
    {
        bodyForce_parallel(p, dt, nBodies, number_of_threads);

        saveToCSV(p, nBodies, iter, folder);

        for (int i = 0; i < nBodies; i++)
        {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    free(fakeBuf);

    Result *result = malloc(sizeof(Result));
    if (result)
    {
        result->execution_time = execution_time;
        result->number_of_threads = number_of_threads;
        result->all_bodies = p;
    }
    return result;
}

int are_results_equal(Result *sequential_result, Result *parallel_result, int n_bodies)
{
    Body *seq = sequential_result->all_bodies,
         *par = parallel_result->all_bodies;
    for (int i = 0; i < n_bodies; i++)
    {
        if ((seq[i].x < (par[i].x - ACCURACY)) || (seq[i].x > (par[i].x + ACCURACY)) ||
            (seq[i].y < (par[i].y - ACCURACY)) || (seq[i].y > (par[i].y + ACCURACY)) ||
            (seq[i].z < (par[i].z - ACCURACY)) || (seq[i].z > (par[i].z + ACCURACY)) ||
            (seq[i].vx < (par[i].vx - ACCURACY)) || (seq[i].vx > (par[i].vx + ACCURACY)) ||
            (seq[i].vy < (par[i].vy - ACCURACY)) || (seq[i].vy > (par[i].vy + ACCURACY)) ||
            (seq[i].vz < (par[i].vz - ACCURACY)) || (seq[i].vz > (par[i].vz + ACCURACY)))
        {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    int nBodies = atoi(argv[1]);
    int bytes = nBodies * sizeof(Body);
    float *buf = (float *)malloc(bytes);
    randomizeBodies(buf, 6 * nBodies);

    float **copies = (float **)malloc(4 * sizeof(float *));
    for (int i = 0; i < 4; i++)
    {
        copies[i] = (float *)malloc(bytes);
        for (int j = 0; j < bytes / sizeof(float); j++)
        {
            copies[i][j] = buf[j];
        }
    }

    Result *sequential_result = sequential_implementation(argv, buf);
    printf("Sequential implementation execution time: %fs\n", sequential_result->execution_time);

    Result *parallel_result_one_thread = parallel_implementation(argv, 1, copies[0]);
    printf("Parallel implementation (one thread) execution time: %fs\n", parallel_result_one_thread->execution_time);
    if (are_results_equal(sequential_result, parallel_result_one_thread, nBodies))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    Result *parallel_result_two_threads = parallel_implementation(argv, 2, copies[1]);
    printf("Parallel implementation (two threads) execution time: %fs\n", parallel_result_two_threads->execution_time);
    if (are_results_equal(sequential_result, parallel_result_two_threads, nBodies))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    Result *parallel_result_four_threads = parallel_implementation(argv, 4, copies[2]);
    printf("Parallel implementation (four threads) execution time: %fs\n", parallel_result_four_threads->execution_time);
    if (are_results_equal(sequential_result, parallel_result_four_threads, nBodies))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    Result *parallel_result_eight_threads = parallel_implementation(argv, 8, copies[3]);
    printf("Parallel implementation (eight threads) execution time: %fs\n", parallel_result_eight_threads->execution_time);
    if (are_results_equal(sequential_result, parallel_result_eight_threads, nBodies))
        printf("Test PASSED\n\n");
    else
        printf("Test FAILED\n\n");

    free(buf);
    for (int i = 0; i < 4; i++)
    {
        free(copies[i]);
    }
    free(copies);

    free(sequential_result);
    free(parallel_result_one_thread);
    free(parallel_result_two_threads);
    free(parallel_result_four_threads);
    free(parallel_result_eight_threads);
}
