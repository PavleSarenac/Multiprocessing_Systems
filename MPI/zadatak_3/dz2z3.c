#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <mpi.h>
#include <string.h>

#define SOFTENING 1e-9f
#define MASTER 0
#define ACCURACY 0.01f
#define N 8
#define TOO_MANY_PROCESSES 1

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

typedef struct Result
{
    double executionTime;
    Body *bodies;
} Result;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void bodyForceSequential(Body *p, float dt, int n)
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

void bodyForceParallel(Body *p, float dt, int n, int start, int end)
{
    for (int i = start; i < end; i++)
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

void sequentialImplementation(int argc, char **argv, float *initialBuf, Result *sequentialResult)
{
    double startTime = MPI_Wtime();

    int nBodies = atoi(argv[1]);
    int nIters = atoi(argv[2]);
    const char *folder = argv[3];

    const float dt = 0.01f;

    mkdir(folder, 0700);

    int bytes = nBodies * sizeof(Body);
    float *buf = (float *)malloc(bytes);
    memcpy(buf, initialBuf, bytes);
    float *fakeBuf = (float *)malloc(bytes);
    Body *p = (Body *)buf;

    randomizeBodies(fakeBuf, 6 * nBodies);

    for (int iter = 0; iter < nIters; iter++)
    {

        bodyForceSequential(p, dt, nBodies);

        saveToCSV(p, nBodies, iter, folder);

        for (int i = 0; i < nBodies; i++)
        {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    }

    free(fakeBuf);

    double endTime = MPI_Wtime();
    double executionTime = endTime - startTime;

    sequentialResult->bodies = p;
    sequentialResult->executionTime = executionTime;
}

void parallelImplementation(int argc, char **argv, float *initialBuf, Result *parallelResult)
{
    double startTime = MPI_Wtime();

    int nBodies, nIters;
    const char *folder;

    const float dt = 0.01f;

    int processRank, communicatorSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

    if (communicatorSize > N)
    {
        MPI_Abort(MPI_COMM_WORLD, TOO_MANY_PROCESSES);
    }

    if (processRank == MASTER)
    {
        nBodies = atoi(argv[1]);
        nIters = atoi(argv[2]);
        folder = argv[3];
        mkdir(folder, 0700);
    }

    MPI_Bcast(&nBodies, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&nIters, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    int bytes = nBodies * sizeof(Body);
    float *buf = (float *)malloc(bytes);
    memcpy(buf, initialBuf, bytes);
    float *fakeBuf = (float *)malloc(bytes);
    float *recvBuf = (float *)malloc(bytes);
    Body *p = (Body *)buf;

    if (processRank == MASTER)
    {
        randomizeBodies(fakeBuf, 6 * nBodies);
    }

    int chunkSize = (nBodies + communicatorSize - 1) / communicatorSize;
    int start = processRank * chunkSize;
    int end = (start + chunkSize < nBodies ? start + chunkSize : nBodies);

    int shouldBeIncluded;
    if (end > start)
        shouldBeIncluded = 1;
    else
        shouldBeIncluded = 0;

    MPI_Comm reducedComm;
    MPI_Comm_split(MPI_COMM_WORLD, shouldBeIncluded, processRank, &reducedComm);
    MPI_Comm_size(reducedComm, &communicatorSize);

    if (end > start)
    {
        int recvCounts[communicatorSize];
        int displacements[communicatorSize];
        for (int i = 0; i < communicatorSize; i++)
        {
            displacements[i] = i * chunkSize * 6;
            if (i < communicatorSize - 1)
            {
                recvCounts[i] = chunkSize * 6;
            }
            else
            {
                recvCounts[i] = (nBodies - ((communicatorSize - 1) * chunkSize)) * 6;
            }
        }

        for (int iter = 0; iter < nIters; iter++)
        {
            bodyForceParallel(p, dt, nBodies, start, end);

            MPI_Allgatherv(buf + start * 6, (end - start) * 6, MPI_FLOAT,
                           recvBuf, recvCounts, displacements, MPI_FLOAT, reducedComm);

            memcpy(buf, recvBuf, nBodies * 6 * sizeof(float));

            if (processRank == MASTER)
            {
                saveToCSV(p, nBodies, iter, folder);
            }

            for (int i = 0; i < nBodies; i++)
            {
                p[i].x += p[i].vx * dt;
                p[i].y += p[i].vy * dt;
                p[i].z += p[i].vz * dt;
            }
        }
    }

    free(fakeBuf);
    free(recvBuf);
    if (processRank == MASTER)
    {
        double endTime = MPI_Wtime();
        double executionTime = endTime - startTime;

        parallelResult->bodies = p;
        parallelResult->executionTime = executionTime;
    }
}

int areResultsEqual(Result *sequentialResult, Result *parallelResult, int nBodies)
{
    Body *seq = sequentialResult->bodies,
         *par = parallelResult->bodies;
    for (int i = 0; i < nBodies; i++)
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
    MPI_Init(&argc, &argv);
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    Result *sequentialResult, *parallelResult = malloc(sizeof(Result));

    int nBodies = atoi(argv[1]);
    int bytes = nBodies * sizeof(Body);
    float *initialBuf = (float *)malloc(bytes);

    if (processRank == MASTER)
    {
        sequentialResult = malloc(sizeof(Result));
        randomizeBodies(initialBuf, nBodies * 6);
    }

    MPI_Bcast(initialBuf, nBodies * 6, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    if (processRank == MASTER)
    {
        /*
         * The main purpose of executing sequential implementation here is to retrieve its results so
         * that they can be compared with the results of the parallel implementation. The execution time
         * of the sequential implementation here is not relevant because it is executed inside of the MPI
         * world, so it is much slower than the actual sequential implementation executed outside of the
         * MPI world.
         */
        sequentialImplementation(argc, argv, initialBuf, sequentialResult);
        printf("Sequential implementation execution time: %fs\n", sequentialResult->executionTime);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    parallelImplementation(argc, argv, initialBuf, parallelResult);
    if (processRank == MASTER)
    {
        printf("Parallel implementation execution time: %fs\n", parallelResult->executionTime);
        if (areResultsEqual(sequentialResult, parallelResult, nBodies))
            printf("Test PASSED\n");
        else
            printf("Test FAILED\n");
    }

    free(initialBuf);
    free(parallelResult->bodies);
    if (processRank == MASTER)
    {
        free(sequentialResult->bodies);
        free(sequentialResult);
        free(parallelResult);
    }
    MPI_Finalize();
    return 0;
}