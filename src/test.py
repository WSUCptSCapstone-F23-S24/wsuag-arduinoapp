import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO('detect-plot.pt')  # Use the correct path to your model file

# Directory containing the images
image_dir = 'testset/s2'  # Adjust this to your specific directory

# List all image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Run batched inference on all images in the folder
results = model.predict(source=image_files, save=True, save_txt=True)

# Process results for each image
for result in results:
    # Assuming results has masks as numpy arrays
    masks = result.masks  # Masks object for segmentation masks outputs

    # Erosion settings
    kernel = np.ones((5, 5), np.uint8)  # Define the kernel size, adjust as needed
    iterations = 1  # Define the number of erosion iterations, adjust as needed

    # Check and process each mask
    for i, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)  # Convert mask to numpy array if not already
        eroded_mask = cv2.erode(mask, kernel, iterations=iterations)  # Apply erosion

        output_path = os.path.join(image_dir, f'eroded_mask_{i}.png')  # Define output path
        cv2.imwrite(output_path, eroded_mask)  # Save the eroded mask as an image

    print(f'Processed image: {result.files}')  # Example of accessing result file names
