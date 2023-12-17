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
#define INVALID_NUMBER_OF_PROCESSES 1

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

enum Tags
{
    BUF_TAG = 1000,
    NBODIES_TAG,
    START_INDEX_TAG,
    END_INDEX_TAG,
    RESULT_TAG,
    SHOULD_KEEP_WORKING_TAG
};

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

int main(int argc, char **argv)
{
    int nBodies, nIters;
    const char *folder;

    const float dt = 0.01f;

    int processRank, communicatorSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

    double startTime = MPI_Wtime();

    if (communicatorSize < 2 || communicatorSize > N)
    {
        MPI_Abort(MPI_COMM_WORLD, INVALID_NUMBER_OF_PROCESSES);
    }

    if (processRank == MASTER)
    {
        nBodies = atoi(argv[1]);
        nIters = atoi(argv[2]);
        folder = argv[3];
        mkdir(folder, 0700);
    }

    MPI_Bcast(&nBodies, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    int bytes = nBodies * sizeof(Body);
    float *buf = (float *)malloc(bytes);
    float *recvBuf;
    Body *p = (Body *)buf;

    if (processRank == MASTER)
    {
        randomizeBodies(buf, 6 * nBodies);
    }

    MPI_Bcast(buf, nBodies * 6, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    int chunkSize, start, end;
    int keepOnWorking;
    int workingProcesses;

    if (processRank == MASTER)
    {
        chunkSize = (nBodies + (communicatorSize - 1) - 1) / (communicatorSize - 1);
        workingProcesses = 0;
        MPI_Request request;
        for (int currentProcessRank = 1; currentProcessRank < communicatorSize; currentProcessRank++)
        {
            start = (currentProcessRank - 1) * chunkSize;
            end = (start + chunkSize < nBodies ? start + chunkSize : nBodies);
            MPI_Isend(&start, 1, MPI_INT, currentProcessRank, START_INDEX_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&end, 1, MPI_INT, currentProcessRank, END_INDEX_TAG, MPI_COMM_WORLD, &request);
            if (end > start)
            {
                workingProcesses++;
            }
        }

        for (int iter = 0; iter < nIters; iter++)
        {
            if (iter > 0)
            {
                for (int currentProcessRank = 1; currentProcessRank <= workingProcesses; currentProcessRank++)
                {
                    keepOnWorking = 1;
                    MPI_Isend(&keepOnWorking, 1, MPI_INT, currentProcessRank,
                              SHOULD_KEEP_WORKING_TAG, MPI_COMM_WORLD, &request);
                    MPI_Isend(buf, nBodies * 6, MPI_FLOAT, currentProcessRank,
                              BUF_TAG, MPI_COMM_WORLD, &request);
                }
            }

            for (int currentProcessRank = 1; currentProcessRank <= workingProcesses; currentProcessRank++)
            {
                MPI_Status status;
                int recvCount;
                if (currentProcessRank != communicatorSize - 1)
                {
                    recvCount = chunkSize * 6;
                }
                else
                {
                    recvCount = (nBodies - ((workingProcesses - 1) * chunkSize)) * 6;
                }
                int bufOffset = (currentProcessRank - 1) * chunkSize * 6;
                MPI_Recv(buf + bufOffset, recvCount, MPI_FLOAT, currentProcessRank,
                         RESULT_TAG, MPI_COMM_WORLD, &status);
            }

            saveToCSV(p, nBodies, iter, folder);
        }

        for (int currentProcessRank = 1; currentProcessRank <= workingProcesses; currentProcessRank++)
        {
            keepOnWorking = 0;
            MPI_Isend(&keepOnWorking, 1, MPI_INT, currentProcessRank,
                      SHOULD_KEEP_WORKING_TAG, MPI_COMM_WORLD, &request);
        }
    }
    else
    {
        MPI_Status status;
        MPI_Request request;

        MPI_Recv(&start, 1, MPI_INT, MASTER, START_INDEX_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&end, 1, MPI_INT, MASTER, END_INDEX_TAG, MPI_COMM_WORLD, &status);

        if (end > start)
        {
            bodyForceParallel(p, dt, nBodies, start, end);
            MPI_Isend(buf + start * 6, (end - start) * 6, MPI_FLOAT, MASTER, RESULT_TAG, MPI_COMM_WORLD, &request);
            MPI_Recv(&keepOnWorking, 1, MPI_INT, MASTER, SHOULD_KEEP_WORKING_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(buf, nBodies * 6, MPI_FLOAT, MASTER, BUF_TAG, MPI_COMM_WORLD, &status);
            while (keepOnWorking)
            {
                for (int i = 0; i < nBodies; i++)
                {
                    p[i].x += p[i].vx * dt;
                    p[i].y += p[i].vy * dt;
                    p[i].z += p[i].vz * dt;
                }
                bodyForceParallel(p, dt, nBodies, start, end);
                MPI_Isend(buf + start * 6, (end - start) * 6, MPI_FLOAT, MASTER, RESULT_TAG, MPI_COMM_WORLD,
                          &request);
                MPI_Recv(&keepOnWorking, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (keepOnWorking)
                {
                    MPI_Recv(buf, nBodies * 6, MPI_FLOAT, MASTER, BUF_TAG, MPI_COMM_WORLD, &status);
                }
            }
        }
    }

    free(buf);

    if (processRank == MASTER)
    {
        double endTime = MPI_Wtime();
        printf("Parallel implementation execution time: %fs\n", endTime - startTime);
    }

    MPI_Finalize();
    return 0;
}
