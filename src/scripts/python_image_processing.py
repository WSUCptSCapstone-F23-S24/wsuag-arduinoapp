import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def make_constant_csv(in_path):
    # Ensure the input path exists
    if not os.path.exists(in_path):
        print(f"Error: The input path '{in_path}' does not exist.")
        return

    # Create an empty list to collect RGB data
    sp_data = []

    # List only JPEG files in the input path
    r_in_data = [img for img in os.listdir(in_path) if img.endswith('.jpg')]

    if not r_in_data:
        print("No JPEG files found in the input path.")
        return

    print('start')

    # Loop to extract the RGB data
    for file in r_in_data:
        img = cv2.imread(os.path.join(in_path, file))  # Read the image

        if img is None:
            print(f"Error: Unable to read the image '{file}'. Skipping...")
            continue

        # Split the color channels
        b, g, r = cv2.split(img)

        # Set list of data
        sp = [b, g, r]
        color = ['blue', 'green', 'red']

        # Create an empty list to collect data of each spectrum in the loop
        cl = []

        # For loop to extract the RGB data
        for s, c in zip(sp, color):
            # Extract statistical data
            mean = round(np.nanmean(s), 5)
            median = round(np.nanmedian(s), 5)
            std = round(np.nanstd(s), 5)
            max_value = round(np.nanmax(s), 5)
            p95 = round(np.nanpercentile(s, 95), 5)
            p90 = round(np.nanpercentile(s, 90), 5)
            p85 = round(np.nanpercentile(s, 85), 5)

            # Extract date and time information from the file name
            parts = file.split('_')
            if len(parts) >= 2:
                date = parts[0]
                time = parts[1]
                date_time = date[:5] + '_' + time[:2]
            else:
                date, time, date_time = "", "", ""

            # Extract representation picture information from the file name
            if len(parts) > 2:
                rep_pic0 = parts[-1]
                rep_pic = rep_pic0.split('.')[0]
            else:
                rep_pic = ""

            spectrum = c

            # Make a dictionary of extracted data for one color
            data = [date, time, date_time, spectrum, mean, median, std, max_value, p95, p90, p85, rep_pic]
            cl.append(data)

        # Combine data from all spectra
        sp_data.extend(cl)

    if sp_data:
        # Make a DataFrame to save as a CSV file
        header = ['date', 'time', 'date_time', 'spectrum', 'mean', 'median', 'std', 'max', 'p95', 'p90', 'p85', 'rep_pic']
        df_final = pd.DataFrame(sp_data, columns=header)

        # Save the DataFrame to a CSV file
        csv_filename = os.path.join(in_path, "test4.csv")
        df_final.to_csv(csv_filename, index=False)
        print(f"CSV file saved as '{csv_filename}'")

    print('finish')

# Example usage:
if __name__ == "__main__":
    input_path = "C:\\Users\\shrey\\Desktop\\421sprint2\\wsuag-arduinoapp\\src\\tf2.0\\models\\research\\object_detection\\test_images\\crop_test"
    make_constant_csv(input_path)