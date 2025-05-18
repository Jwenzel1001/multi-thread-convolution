#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void apply_sobel_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height);
void apply_prewitt_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height);
void save_image(uint8_t *output, const char *filename, int width, int height);

int main(int argc, char *argv[]) {
    omp_set_num_threads(4);
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_folder> <width> <height>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_folder = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    // Construct the output file paths
    char sobel_output_file[512], prewitt_output_file[512];
    snprintf(sobel_output_file, sizeof(sobel_output_file), "%s/sobel_output.bin", output_folder);
    snprintf(prewitt_output_file, sizeof(prewitt_output_file), "%s/prewitt_output.bin", output_folder);

    // Open the input binary file for reading
    FILE *file = fopen(input_file, "rb");
    if (!file) {
        perror("Failed to open input file");
        return 1;
    }

    // Allocate memory for the flattened RGB image and read data
    uint8_t *image = (uint8_t *)malloc(height * width * 3 * sizeof(uint8_t));
    fread(image, sizeof(uint8_t), height * width * 3, file);
    fclose(file);

    // Allocate memory for the flattened output images
    uint8_t *sobel_output = (uint8_t *)calloc(height * width * 3, sizeof(uint8_t));
    uint8_t *prewitt_output = (uint8_t *)calloc(height * width * 3, sizeof(uint8_t));

    // Apply Sobel and Prewitt filters in parallel using OpenMP sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            apply_sobel_filter_color_coded(image, sobel_output, width, height);
        }

        #pragma omp section
        {
            apply_prewitt_filter_color_coded(image, prewitt_output, width, height);
        }
    }

    // Save the output images
    save_image(sobel_output, sobel_output_file, width, height);
    save_image(prewitt_output, prewitt_output_file, width, height);

    // Free memory
    free(image);
    free(sobel_output);
    free(prewitt_output);

    return 0;
}

// Function to save an image to a binary file
void save_image(uint8_t *output, const char *filename, int width, int height) {
    FILE *output_file_ptr = fopen(filename, "wb");
    if (!output_file_ptr) {
        perror("Failed to open output file");
        exit(1);
    }
    fwrite(output, sizeof(uint8_t), height * width * 3, output_file_ptr);
    fclose(output_file_ptr);
    printf("Filter applied and result saved to '%s'.\n", filename);
}

void apply_sobel_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; y++) {
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
            int idx = (y * width + x) * 3;
            output[idx] = abs(sumX) > 255 ? 255 : abs(sumX); // Red channel for horizontal edges
            output[idx + 2] = abs(sumY) > 255 ? 255 : abs(sumY); // Blue channel for vertical edges
        }
    }
}

void apply_prewitt_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height) {
    int Px[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int Py[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int idx = ((y + i) * width + (x + j)) * 3;
                    int intensity = (image[idx] + image[idx + 1] + image[idx + 2]) / 3;
                    sumX += intensity * Px[i + 1][j + 1];
                    sumY += intensity * Py[i + 1][j + 1];
                }
            }
            int idx = (y * width + x) * 3;
            output[idx] = abs(sumX) > 255 ? 255 : abs(sumX); // Red channel for horizontal edges
            output[idx + 2] = abs(sumY) > 255 ? 255 : abs(sumY); // Blue channel for vertical edges
        }
    }
}
