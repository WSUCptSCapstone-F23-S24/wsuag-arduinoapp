import cv2
import numpy as np
import os
import argparse

# Function to calculate NDVI statistics from a set of images and save them to a text file
def calculate_ndvi_constants(image_folder):
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')]

    # Initialize a dictionary to store reference NDVI statistics
    reference_ndvi_constants = {
        'mean_ndvi': [],
        'median_ndvi': [],
        'std_ndvi': [],
        'max_ndvi': [],
        'p95_ndvi': [],
        'p90_ndvi': [],
        'p85_ndvi': []
    }

    print('start')

    for image_path in image_paths:
        img = cv2.imread(image_path)

        # Calculate NDVI
        b, g, r = cv2.split(img)
        ndvi = ((1.664 * b.astype(float)) / (0.953 * r.astype(float))) - 1

        # Calculate NDVI statistics for the entire image
        mean_ndvi = round(np.nanmean(ndvi), 5)
        median_ndvi = round(np.nanmedian(ndvi), 5)
        std_ndvi = round(np.nanstd(ndvi), 5)
        max_ndvi = round(np.nanmax(ndvi), 5)
        p95_ndvi = round(np.nanpercentile(ndvi, 95), 5)
        p90_ndvi = round(np.nanpercentile(ndvi, 90), 5)
        p85_ndvi = round(np.nanpercentile(ndvi, 85), 5)

        reference_ndvi_constants['mean_ndvi'].append(mean_ndvi)
        reference_ndvi_constants['median_ndvi'].append(median_ndvi)
        reference_ndvi_constants['std_ndvi'].append(std_ndvi)
        reference_ndvi_constants['max_ndvi'].append(max_ndvi)
        reference_ndvi_constants['p95_ndvi'].append(p95_ndvi)
        reference_ndvi_constants['p90_ndvi'].append(p90_ndvi)
        reference_ndvi_constants['p85_ndvi'].append(p85_ndvi)

    # Calculate the average of NDVI values
    for stat in reference_ndvi_constants:
        reference_ndvi_constants[stat] = np.mean(reference_ndvi_constants[stat])

    # Create and write the NDVI constants to a text file
    output_file = os.path.join(image_folder, 'ndvi_constants.txt')
    with open(output_file, 'w') as file:
        for key, value in reference_ndvi_constants.items():
            file.write(f"{key}: {value}\n")

    return reference_ndvi_constants

def calculate_rgb_constants(image_folder):
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')]

    # Initialize a dictionary to store RGB statistics
    reference_rgb_constants = {
        'mean_r': [],
        'mean_g': [],
        'mean_b': [],
    }

    print('start')

    for image_path in image_paths:
        img = cv2.imread(image_path)

        # Calculate RGB statistics for the entire image
        mean_b = round(np.mean(img[:, :, 0]), 5)
        mean_g = round(np.mean(img[:, :, 1]), 5)
        mean_r = round(np.mean(img[:, :, 2]), 5)

        reference_rgb_constants['mean_b'].append(mean_b)
        reference_rgb_constants['mean_g'].append(mean_g)
        reference_rgb_constants['mean_r'].append(mean_r)

    # Calculate the average of RGB values
    for stat in reference_rgb_constants:
        reference_rgb_constants[stat] = np.mean(reference_rgb_constants[stat])

    # Create and write the RGB constants to a text file
    output_file = os.path.join(image_folder, 'rgb_constants.txt')
    with open(output_file, 'w') as file:
        for key, value in reference_rgb_constants.items():
            file.write(f"{key}: {value}\n")

    return reference_rgb_constants

# Function to normalize an image based on reference panel and NDVI constants
def normalize_image(image_path, reference_panel_coords, constants):
    # Load the image you want to normalize
    image = cv2.imread(image_path)

    # Crop the reference panel from the image using the specified coordinates
    x1, y1, x2, y2 = reference_panel_coords
    reference_panel = image[y1:y2, x1:x2]

    # Calculate the mean color values of the reference panel for each channel
    b_mean = np.mean(reference_panel[:, :, 0])
    g_mean = np.mean(reference_panel[:, :, 1])
    r_mean = np.mean(reference_panel[:, :, 2])

    # Calculate correction factors based on the reference panel and constants
    brightness_correction = (b_mean / constants['mean_ndvi'], g_mean / constants['mean_ndvi'], r_mean / constants['mean_ndvi'])
    contrast_correction = (constants['std_ndvi'] / 0.045, constants['std_ndvi'] / 0.045, constants['std_ndvi'] / 0.045)

    # Apply the brightness and contrast correction to the entire image
    normalized_image = cv2.convertScaleAbs(image, alpha=contrast_correction, beta=brightness_correction)

    # Save or display the normalized image
    cv2.imwrite('normalized_image.jpg', normalized_image)
    cv2.imshow('Normalized Image', normalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize_images_in_folder(input_folder, rgb_file):
    # Read the RGB correction values from the file
    print(input_folder)
    print(rgb_file)
    with open(rgb_file, 'r') as file:
        rgb_values = file.read().split()
        blue_correction = float(rgb_values[0]) /100
        green_correction = float(rgb_values[1]) / 100
        red_correction = float(rgb_values[2]) / 100
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image_name, _ = os.path.splitext(filename)

            # Load the image you want to normalize
            image = cv2.imread(image_path)

            # Apply the RGB correction factors to the entire image
            normalized_image = cv2.merge([
                (image[:, :, 0] * blue_correction).astype('uint8'),
                (image[:, :, 1] * green_correction).astype('uint8'),
                (image[:, :, 2] * red_correction).astype('uint8')
            ])

            # Determine the output file path in the same location as the original image
            output_folder = os.path.dirname(image_path)
            output_file = os.path.join(input_folder, f'{image_name}_normalized.jpg')
            print(image_name)
            # Save the normalized image with the original image's name and "_normalized" suffix
            cv2.imwrite(output_file, normalized_image)

def main():
    parser = argparse.ArgumentParser(description="Normalize images using RGB correction values")
    parser.add_argument("input_folder", help="Path to the folder containing images")
    parser.add_argument("rgb_file", help="Path to the file containing RGB correction values")

    args = parser.parse_args()
    normalize_images_in_folder(args.input_folder, args.rgb_file)

if __name__ == '__main__':
    main()


# python image_normalization.py .\test_images\crop_test
# python image_normalization.py .\test_images\image_correct_test .\output_images\box_1\constants.txt