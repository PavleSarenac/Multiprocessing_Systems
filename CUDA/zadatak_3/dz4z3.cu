#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#define SOFTENING 1e-9f
#define ACCURACY 0.01f
#define NUMBER_OF_THREADS_PER_BLOCK 256

typedef struct
{
    float execution_time;
    float *coordinates;
} Result;

__host__ void randomizeBodies(float *data, int n)
{
    int nBodies = n / 6;
    for (int i = 0; i < nBodies; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            data[j * nBodies + i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        }
    }
}

__host__ void bodyForce(float *buf, float dt, int n)
{
    for (int i = 0; i < n; i++)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            float dx = buf[j] - buf[i];
            float dy = buf[n + j] - buf[n + i];
            float dz = buf[2 * n + j] - buf[2 * n + i];
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        buf[3 * n + i] += dt * Fx;
        buf[4 * n + i] += dt * Fy;
        buf[5 * n + i] += dt * Fz;
    }
}

__global__ void bodyForceKernel(float *buf_gpu, int nBodies, float dt)
{
    extern __shared__ float sharedMemory[];
    float *allPositions = sharedMemory;
    float *velocities = allPositions + nBodies * 3;

    // Loading allPositions from global memory into shared memory.
    int positionsChunkSize = (nBodies * 3 + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK;
    int positionsStart = threadIdx.x * positionsChunkSize;
    int positionsEnd;
    if (positionsStart + positionsChunkSize < nBodies * 3)
    {
        positionsEnd = positionsStart + positionsChunkSize;
    }
    else
    {
        positionsEnd = nBodies * 3;
    }

    for (int i = positionsStart; i < positionsEnd; i++)
    {
        allPositions[i] = buf_gpu[i];
    }

    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Loading velocities from global memory into shared memory.
    // Load vx.
    velocities[threadIdx.x] = buf_gpu[nBodies * 3 + globalThreadId];
    // Load vy.
    velocities[NUMBER_OF_THREADS_PER_BLOCK + threadIdx.x] = buf_gpu[nBodies * 4 + globalThreadId];
    // Load vz.
    velocities[2 * NUMBER_OF_THREADS_PER_BLOCK + threadIdx.x] = buf_gpu[nBodies * 5 + globalThreadId];

    __syncthreads();

    // Calculate new velocities.
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < nBodies; j++)
    {
        float dx = allPositions[j] - allPositions[globalThreadId];
        float dy = allPositions[nBodies + j] - allPositions[nBodies + globalThreadId];
        float dz = allPositions[2 * nBodies + j] - allPositions[2 * nBodies + globalThreadId];
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    velocities[threadIdx.x] += dt * Fx;
    velocities[NUMBER_OF_THREADS_PER_BLOCK + threadIdx.x] += dt * Fy;
    velocities[2 * NUMBER_OF_THREADS_PER_BLOCK + threadIdx.x] += dt * Fz;

    __syncthreads();

    buf_gpu[nBodies * 3 + globalThreadId] = velocities[threadIdx.x];
    buf_gpu[nBodies * 4 + globalThreadId] = velocities[NUMBER_OF_THREADS_PER_BLOCK + threadIdx.x];
    buf_gpu[nBodies * 5 + globalThreadId] = velocities[2 * NUMBER_OF_THREADS_PER_BLOCK + threadIdx.x];
}

__host__ void saveToCSV(float *buf, int n, int iter, const char *folder)
{
    char filename[50];
    sprintf(filename, "%s/iteration_%d.csv", folder, iter);
    FILE *file = fopen(filename, "w");

    fprintf(file, "x,y,z,vx,vy,vz\n");
    for (int i = 0; i < n; i++)
    {
        fprintf(file, "%f,%f,%f,%f,%f,%f\n", buf[i], buf[n + i], buf[2 * n + i], buf[3 * n + i], buf[4 * n + i],
                buf[5 * n + i]);
    }

    fclose(file);
}

// buf array is organized like this in memory (when nBodies == 2): x,x,y,y,z,z,vx,vx,vy,vy,vz,vz

// It is implemented like that because we want GPU threads within a warp to access consecutive memory locations so that
// bank conflicts don't happen, and to use burst sections from GPU memory to its fullest potential.

__host__ Result *nbodyMainCPU(char **argv, float *buf)
{
    Result *sequential_result = (Result *)malloc(sizeof(Result));

    int nBodies = atoi(argv[1]);
    int nIters = atoi(argv[2]);
    const char *folder = argv[3];

    const float dt = 0.01f;

    mkdir(folder, 0700);

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int iter = 0; iter < nIters; iter++)
    {
        bodyForce(buf, dt, nBodies);

        saveToCSV(buf, nBodies, iter, folder);

        for (int i = 0; i < nBodies; i++)
        {
            buf[i] += buf[3 * nBodies + i] * dt;
            buf[nBodies + i] += buf[4 * nBodies + i] * dt;
            buf[2 * nBodies + i] += buf[5 * nBodies + i] * dt;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    sequential_result->execution_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    sequential_result->coordinates = buf;

    return sequential_result;
}

__host__ Result *nbodyMainGPU(char **argv, float *buf)
{
    Result *parallel_result = (Result *)malloc(sizeof(Result));

    int nBodies = atoi(argv[1]);
    int nIters = atoi(argv[2]);
    const char *folder = argv[3];

    const float dt = 0.01f;

    mkdir(folder, 0700);

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    dim3 gridDimension((nBodies + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK);
    dim3 blockDimension(NUMBER_OF_THREADS_PER_BLOCK);

    float *buf_gpu;
    cudaMalloc(&buf_gpu, nBodies * 6 * sizeof(float));
    cudaMemcpy(buf_gpu, buf, nBodies * 6 * sizeof(float), cudaMemcpyHostToDevice);
    int sharedMemorySize = nBodies * 3 * sizeof(float) + NUMBER_OF_THREADS_PER_BLOCK * 3 * sizeof(float);

    for (int iter = 0; iter < nIters; iter++)
    {
        bodyForceKernel<<<gridDimension, blockDimension, sharedMemorySize>>>(buf_gpu, nBodies, dt);
        cudaMemcpy(buf, buf_gpu, nBodies * 6 * sizeof(float), cudaMemcpyDeviceToHost);

        saveToCSV(buf, nBodies, iter, folder);

        for (int i = 0; i < nBodies; i++)
        {
            buf[i] += buf[3 * nBodies + i] * dt;
            buf[nBodies + i] += buf[4 * nBodies + i] * dt;
            buf[2 * nBodies + i] += buf[5 * nBodies + i] * dt;
        }

        cudaMemcpy(buf_gpu, buf, nBodies * 6 * sizeof(float), cudaMemcpyHostToDevice);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    parallel_result->execution_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    parallel_result->coordinates = buf;

    return parallel_result;
}

int are_results_equal(Result *sequential_result, Result *parallel_result, int nBodies)
{
    float *seq = sequential_result->coordinates,
          *par = parallel_result->coordinates;
    for (int i = 0; i < nBodies * 6; i++)
    {
        if ((seq[i] < (par[i] - ACCURACY)) || (seq[i] > (par[i] + ACCURACY)))
        {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    Result *sequential_result, *parallel_result;

    int nBodies = atoi(argv[1]);
    int bytes = nBodies * 6 * sizeof(float);
    float *buf = (float *)malloc(bytes);
    randomizeBodies(buf, 6 * nBodies);

    float **copies = (float **)malloc(4 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        copies[i] = (float *)malloc(bytes);
        for (int j = 0; j < bytes / sizeof(float); j++)
        {
            copies[i][j] = buf[j];
        }
    }

    sequential_result = nbodyMainCPU(argv, copies[0]);
    parallel_result = nbodyMainGPU(argv, copies[1]);

    printf("Sequential implementation execution time: %fs\n", sequential_result->execution_time);
    printf("Parallel implementation execution time: %fs\n", parallel_result->execution_time);
    if (are_results_equal(sequential_result, parallel_result, nBodies))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    free(sequential_result->coordinates);
    free(sequential_result);

    free(parallel_result->coordinates);
    free(parallel_result);

    free(buf);

    return 0;
}
