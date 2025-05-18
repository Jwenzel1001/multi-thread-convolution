#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

void apply_sobel_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height);
void apply_prewitt_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height);

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <input_file> <output_folder> <width> <height>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_folder = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);

    // Allocate memory for the flattened image and output arrays
    uint8_t *image = (uint8_t *)malloc(height * width * 3 * sizeof(uint8_t));
    uint8_t *output = (uint8_t *)calloc(height * width * 3, sizeof(uint8_t)); // Zero-initialized output array

    // Read the binary input image file
    FILE *file = fopen(input_file, "rb");
    if (!file) {
        perror("Failed to open input file");
        free(image);
        free(output);
        return 1;
    }
    fread(image, sizeof(uint8_t), height * width * 3, file);
    fclose(file);

    // Apply Sobel filter with color coding
    apply_sobel_filter_color_coded(image, output, width, height);

    // Generate Sobel output path
    char sobel_output_path[256];
    snprintf(sobel_output_path, sizeof(sobel_output_path), "%s/sobel_output.bin", output_folder);

    // Save Sobel output image to binary file
    FILE *output_file = fopen(sobel_output_path, "wb");
    if (!output_file) {
        perror("Failed to open Sobel output file");
        free(image);
        free(output);
        return 1;
    }
    fwrite(output, sizeof(uint8_t), height * width * 3, output_file);
    fclose(output_file);
    printf("Sobel filter applied with color coding and result saved to '%s'.\n", sobel_output_path);

    // Apply Prewitt filter with color coding
    apply_prewitt_filter_color_coded(image, output, width, height);

    // Generate Prewitt output path
    char prewitt_output_path[256];
    snprintf(prewitt_output_path, sizeof(prewitt_output_path), "%s/prewitt_output.bin", output_folder);

    // Save Prewitt output image to binary file
    output_file = fopen(prewitt_output_path, "wb");
    if (!output_file) {
        perror("Failed to open Prewitt output file");
        free(image);
        free(output);
        return 1;
    }
    fwrite(output, sizeof(uint8_t), height * width * 3, output_file);
    fclose(output_file);
    printf("Prewitt filter applied with color coding and result saved to '%s'.\n", prewitt_output_path);

    // Free memory
    free(image);
    free(output);

    return 0;
}

void apply_sobel_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

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
            output[idx + 1] = 0;                            // Green channel unused
            output[idx + 2] = abs(sumY) > 255 ? 255 : abs(sumY); // Blue channel for vertical edges
        }
    }
}

void apply_prewitt_filter_color_coded(uint8_t *image, uint8_t *output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int Gy[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};

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
            output[idx + 1] = 0;                            // Green channel unused
            output[idx + 2] = abs(sumY) > 255 ? 255 : abs(sumY); // Blue channel for vertical edges
        }
    }
}
