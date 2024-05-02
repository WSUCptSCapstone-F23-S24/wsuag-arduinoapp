# Import the necessary function from your module
from utilities import make_constant_csv, slice_into_boxes

from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import math
import random

# Function to process all images in a given folder
def process_images_in_folder(folder_path):
    # Process all images with the 'make_constant_csv' function
    make_constant_csv(folder_path)

    # Assuming you want to also slice each image in the folder
    output_folder = 'output_path_here'  # Specify your output folder
    x, y, length = 100, 100, 50  # Example parameters for the slicing function

    # Loop through each image in the folder
    for image_name in os.listdir(folder_path):
        if image_name.endswith(".png"):  # Check if the file is an image
            image_path = os.path.join(folder_path, image_name)
            
            # Call the slicing function
            slice_into_boxes(image_path, output_folder, x, y, length, render_rectangle=True)

# Path to the folder containing your images
folder_path = "./test_results_color"  # Replace this with your actual folder path

# Run the function to process images in the specified folder
process_images_in_folder(folder_path)
