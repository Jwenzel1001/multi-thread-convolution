#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

// Function declarations
void apply_sobel_filter_color_coded(uint8_t *image, uint8_t *output, int width, int start_row, int end_row, int height);
void apply_prewitt_filter_color_coded(uint8_t *image, uint8_t *output, int width, int start_row, int end_row, int height);

int main(int argc, char *argv[]) {
    double start_time, end_time, io_time = 0.0, computation_time = 0.0, communication_time = 0.0;

    // Start total timer
    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            printf("Usage: %s <input_file> <output_folder> <width> <height>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_folder = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int image_size = width * height * 3;

    uint8_t *image = NULL;

    // Input reading (timed on rank 0)
    if (rank == 0) {
        io_time -= MPI_Wtime();
        image = (uint8_t *)malloc(image_size * sizeof(uint8_t));
        FILE *file = fopen(input_file, "rb");
        if (!file) {
            perror("Failed to open input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fread(image, sizeof(uint8_t), image_size, file);
        fclose(file);
        io_time += MPI_Wtime();
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate rows per process and displacements
    int rows_per_process = height / size;
    int remainder_rows = height % size;

    int *send_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        send_counts[i] = (rows_per_process + (i < remainder_rows ? 1 : 0)) * width * 3;
        displs[i] = (i > 0 ? displs[i - 1] + send_counts[i - 1] : 0);
    }

    // Allocate buffers for local processing
    int local_rows = send_counts[rank] / (width * 3);
    uint8_t *local_image = (uint8_t *)malloc((local_rows + 2) * width * 3 * sizeof(uint8_t)); // +2 for halo rows
    uint8_t *local_sobel_output = (uint8_t *)malloc(local_rows * width * 3 * sizeof(uint8_t));
    uint8_t *local_prewitt_output = (uint8_t *)malloc(local_rows * width * 3 * sizeof(uint8_t));

    // Scatter image data (timed)
    communication_time -= MPI_Wtime();
    MPI_Scatterv(image, send_counts, displs, MPI_UINT8_T, &local_image[width * 3], send_counts[rank], MPI_UINT8_T, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime();

    // Exchange halo rows (timed)
    communication_time -= MPI_Wtime();
    if (rank > 0) { // Send top row to previous rank and receive from it
        MPI_Sendrecv(&local_image[width * 3], width * 3, MPI_UINT8_T, rank - 1, 0,
                     local_image, width * 3, MPI_UINT8_T, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) { // Send bottom row to next rank and receive from it
        MPI_Sendrecv(&local_image[local_rows * width * 3], width * 3, MPI_UINT8_T, rank + 1, 0,
                     &local_image[(local_rows + 1) * width * 3], width * 3, MPI_UINT8_T, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    communication_time += MPI_Wtime();

    // Computation (timed)
    computation_time -= MPI_Wtime();
    apply_sobel_filter_color_coded(local_image, local_sobel_output, width, 1, local_rows + 1, height);
    apply_prewitt_filter_color_coded(local_image, local_prewitt_output, width, 1, local_rows + 1, height);
    computation_time += MPI_Wtime();

    // Gather processed data (timed)
    uint8_t *final_sobel_output = NULL;
    uint8_t *final_prewitt_output = NULL;
    if (rank == 0) {
        final_sobel_output = (uint8_t *)malloc(image_size * sizeof(uint8_t));
        final_prewitt_output = (uint8_t *)malloc(image_size * sizeof(uint8_t));
    }

    communication_time -= MPI_Wtime();
    MPI_Gatherv(local_sobel_output, local_rows * width * 3, MPI_UINT8_T, final_sobel_output, send_counts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_prewitt_output, local_rows * width * 3, MPI_UINT8_T, final_prewitt_output, send_counts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime();

    // Output writing (timed on rank 0)
    if (rank == 0) {
        io_time -= MPI_Wtime();
        char sobel_output_path[256];
        snprintf(sobel_output_path, sizeof(sobel_output_path), "%s/sobel_output.bin", output_folder);
        FILE *output_file = fopen(sobel_output_path, "wb");
        if (!output_file) {
            perror("Failed to open Sobel output file");
        } else {
            fwrite(final_sobel_output, sizeof(uint8_t), image_size, output_file);
            fclose(output_file);
            printf("Sobel filter applied and saved to '%s'.\n", sobel_output_path);
        }

        char prewitt_output_path[256];
        snprintf(prewitt_output_path, sizeof(prewitt_output_path), "%s/prewitt_output.bin", output_folder);
        output_file = fopen(prewitt_output_path, "wb");
        if (!output_file) {
            perror("Failed to open Prewitt output file");
        } else {
            fwrite(final_prewitt_output, sizeof(uint8_t), image_size, output_file);
            fclose(output_file);
            printf("Prewitt filter applied and saved to '%s'.\n", prewitt_output_path);
        }
        io_time += MPI_Wtime();
    }

    // Free memory
    free(local_image);
    free(local_sobel_output);
    free(local_prewitt_output);
    free(send_counts);
    free(displs);
    if (rank == 0) free(image);

    // Finalize MPI and print timing information
    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Total execution time: %.6f seconds\n", end_time - start_time);
        printf("I/O time: %.6f seconds\n", io_time);
        printf("Computation time: %.6f seconds\n", computation_time);
        printf("Communication time: %.6f seconds\n", communication_time);
    }

    MPI_Finalize();
    return 0;
}


// Sobel filter function
void apply_sobel_filter_color_coded(uint8_t *image, uint8_t *output, int width, int start_row, int end_row, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = start_row; y < end_row; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int idx = ((y + i) * width + (x + j)) * 3;
                    int intensity = (image[idx] + image[idx + 1] + image[idx + 2]) / 3;

                    sumX += intensity * Gx[i + 1][j + 1];
                    sumY += intensity * Gy[i + 1][j + 1];
                }
            }

            int idx = ((y - start_row) * width + x) * 3;
            output[idx] = abs(sumX) > 255 ? 255 : abs(sumX);
            output[idx + 1] = 0;
            output[idx + 2] = abs(sumY) > 255 ? 255 : abs(sumY);
        }
    }
}

// Prewitt filter function
void apply_prewitt_filter_color_coded(uint8_t *image, uint8_t *output, int width, int start_row, int end_row, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

    for (int y = start_row; y < end_row; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int idx = ((y + i) * width + (x + j)) * 3;
                    int intensity = (image[idx] + image[idx + 1] + image[idx + 2]) / 3;

                    sumX += intensity * Gx[i + 1][j + 1];
                    sumY += intensity * Gy[i + 1][j + 1];
                }
            }

            int idx = ((y - start_row) * width + x) * 3;
            output[idx] = abs(sumX) > 255 ? 255 : abs(sumX);
            output[idx + 1] = 0;
            output[idx + 2] = abs(sumY) > 255 ? 255 : abs(sumY);
        }
    }
}
