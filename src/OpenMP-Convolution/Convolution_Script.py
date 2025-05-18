import os
import sys
import numpy as np
from PIL import Image
import subprocess
import time

def save_image_as_binary(input_image_path, output_folder, width, height):
    """Converts an image to binary format and saves it."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate the output binary file name
    base_name = os.path.basename(input_image_path)
    file_name, _ = os.path.splitext(base_name)
    output_data_path = os.path.join(output_folder, f"{file_name}_binary.bin")
    
    # Load the image and ensure it's in RGB format
    image = Image.open(input_image_path).convert("RGB")
    image = image.resize((width, height))
    image_data = np.array(image)
    
    # Save the RGB data to a binary file
    image_data.tofile(output_data_path)
    print(f"Image '{input_image_path}' saved as binary data in '{output_data_path}'")
    
    return output_data_path

def run_c_program(binary_path, output_folder, width, height, c_program_path):
    """Calls the C program with binary image path, output folder, width, and height, and tracks execution time."""
    command = f"{c_program_path} {binary_path} {output_folder} {width} {height}"
    print(f"Executing: {command}")
    # Start timing
    start_time = time.time()
    # Run the C program
    result = subprocess.run(command, shell=True)
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    if result.returncode != 0:
        print("Error executing C program.")
    else:
        print("C program executed successfully.")
    # Display the execution time
    print(f"Execution time: {execution_time:.4f} seconds")

def load_and_display_rgb_image(binary_path, width, height, output_folder, filter_name):
    """Loads the processed binary output and saves it as an image with the filter name specified."""
    # Read the binary data and reshape it to an RGB image
    pixel_data = np.fromfile(binary_path, dtype=np.uint8)
    image_data = pixel_data.reshape((height, width, 3))
    
    # Convert to a PIL Image
    output_image = Image.fromarray(image_data, mode='RGB')
    
    # Save and display the image with the filter name in the filename
    output_image_path = os.path.join(output_folder, f"{filter_name}_output.png")
    output_image.save(output_image_path)
    output_image.show()
    print(f"{filter_name.capitalize()} filter applied and saved as '{output_image_path}'")
    return output_image_path

def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_image_path> <output_folder_name> <width> <height>")
        sys.exit(1)

    # Get parameters from command line
    input_image_path = sys.argv[1]
    output_folder_name = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    # Construct the output folder path
    output_folder = f"{output_folder_name}_Output"
    c_program_path = "Convolution_OpenMP.exe"  # Path to compiled C program executable

    # Step 1: Convert image to binary and save it
    binary_path = save_image_as_binary(input_image_path, output_folder, width, height)

    # Step 2: Run the C program with binary input, output folder, width, and height
    run_c_program(binary_path, output_folder, width, height, c_program_path)

    # Step 3: Load and display the processed output
    processed_sobel_path = os.path.join(output_folder, "sobel_output.bin")
    processed_prewitt_path = os.path.join(output_folder, "prewitt_output.bin")
    load_and_display_rgb_image(processed_sobel_path, width, height, output_folder, "sobel")
    load_and_display_rgb_image(processed_prewitt_path, width, height, output_folder, "prewitt")

if __name__ == "__main__":
    main()
