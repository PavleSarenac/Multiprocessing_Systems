#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <mpi.h>
#include <string.h>

#define SOFTENING 1e-9f
#define MASTER 0

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void bodyForce(Body *p, float dt, int n, int start, int end)
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
    float *recvBuf;
    Body *p = (Body *)buf;

    if (processRank == MASTER)
    {
        randomizeBodies(buf, 6 * nBodies);
        recvBuf = (float *)malloc(bytes);
    }

    MPI_Bcast(buf, nBodies * 6, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    int chunkSize = (nBodies + communicatorSize - 1) / communicatorSize;
    int start = processRank * chunkSize;
    int end = (start + chunkSize < nBodies ? start + chunkSize : nBodies);

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
        bodyForce(p, dt, nBodies, start, end);

        MPI_Gatherv(buf + start * 6, (end - start) * 6, MPI_FLOAT,
                    recvBuf, recvCounts, displacements, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

        if (processRank == MASTER)
        {
            memcpy(buf, recvBuf, nBodies * 6 * sizeof(float));

            saveToCSV(p, nBodies, iter, folder);

            for (int i = 0; i < nBodies; i++)
            {
                p[i].x += p[i].vx * dt;
                p[i].y += p[i].vy * dt;
                p[i].z += p[i].vz * dt;
            }
        }

        MPI_Bcast(buf, nBodies * 6, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    }

    free(buf);
    if (processRank == MASTER)
        free(recvBuf);
    MPI_Finalize();
    return 0;
}
