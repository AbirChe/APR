#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define MATRIX_SIZE 4

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int num_procs_per_dimension = sqrt(world_size);
    int block_size = MATRIX_SIZE / num_procs_per_dimension;
    int block_offset_row = (world_rank / num_procs_per_dimension) * block_size;
    int block_offset_col = (world_rank % num_procs_per_dimension) * block_size;

    // Allocation mémoire pour le bloc de lignes et de colonnes de A et le vecteur X
    int* block_A = (int*)malloc(block_size * block_size * sizeof(int));
    int* block_X = (int*)malloc(block_size * sizeof(int));

    // Initialisation du bloc de lignes et de colonnes de A et du vecteur X
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            block_A[i * block_size + j] = (block_offset_row + i + 1) * (block_offset_col + j + 1);
        }
    }
    for (int i = 0; i < block_size; i++) {
        block_X[i] = block_offset_col + i + 1;
    }

    // Collecte de tous les blocs de lignes et de colonnes de A et du vecteur X
    int* all_A = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    int* all_X = (int*)malloc(MATRIX_SIZE * sizeof(int));

    MPI_Allgather(block_A, block_size * block_size, MPI_INT, all_A, block_size * block_size, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(block_X, block_size, MPI_INT, all_X, block_size, MPI_INT, MPI_COMM_WORLD);

    // Calcul de la multiplication matrice-vecteur localement sur chaque processeur
    int* local_result = (int*)malloc(block_size * sizeof(int));
    for (int i = 0; i < block_size; i++) {
        local_result[i] = 0;
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                local_result[i] += all_A[(block_offset_row + i) * MATRIX_SIZE + (block_offset_col + k)] * all_X[(block_offset_col + k)];
            }
        }
    }

    // Collecte des résultats partiels de chaque processeur
    int* all_results = (int*)malloc(MATRIX_SIZE * sizeof(int));
    MPI_Allgather(local_result, block_size, MPI_INT, all_results, block_size, MPI_INT, MPI_COMM_WORLD);

    // Affichage du résultat final sur le processeur 0
    if (world_rank == 0) {
        printf("Résultat : ");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            printf("%d ", all_results[i]);
        }
        printf("\n");
    }

    // Libération de la mémoire
    free(block_A);
    free(block_X);
    free(all_A);
    free(all_X);
    free(local_result);
    free(all_results);

    MPI_Finalize();
    return 0;
}

