import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
import glob
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import pandas as pd
from pathlib import Path
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

#original code by Worasit Sangjan modified by Forest Cook

#plot an array of arrays of 4 tuples: [[(335, 300), (450, 300), (360, 485), (185, 480)]]
#varieties the type of wheat in the plot
#r_b, r_g, r_r are the constant values for the camera
def adjust_images(plots, varieties,r_b,r_g,r_r, in_path):
    # image size
    img_roi = 0

    # Create a list of data in directory
    in_path = in_path
    in_data = [img for img in os.listdir(in_path) if img.endswith('.png')]

    vi = 'ndvi'

    # Image size
    i_w = 1280
    i_h = 1248

    vi_data = []

    print('start')
    # Loop to extract the vi
    for file in in_data:
        img = cv2.imread(os.path.join(in_path, file))  # Read the image
        img = img[img_roi:900, i_w:2496]  # Resize the image

        # Split the color band
        b, g, r = cv2.split(img)
        b = r_b * b
        g = r_g * g
        r = r_r * r

        # Calculate the ndvi
        index = ((1.664 * (b.astype(float))) / (0.953 * (r.astype(float)))) - 1
        # Create black image for masking
        blank = np.zeros(index.shape[:2], dtype='uint8')

        # Mask location on the image

        nd = []

        for pl, var in zip(plots, varieties):
            # Mask the plot in left side
            pl_m = cv2.fillPoly(blank, np.array([pl]), 255)
            m = cv2.bitwise_and(index, index, mask=pl_m)
            m[m <= 0] = np.nan  # Replace zero value to nan
            mean_m = round(np.nanmean(m), 5)
            median_m = round(np.nanmedian(m), 5)
            std_m = round(np.nanstd(m), 5)
            max_m = round(np.nanmax(m), 5)
            p95_m = round(np.nanpercentile(m, 95), 5)
            p90_m = round(np.nanpercentile(m, 90), 5)
            p85_m = round(np.nanpercentile(m, 85), 5)
            date = Path(file[:].split('_')[0]).name
            time = Path(file[:].split('_')[1]).name
            date_time = date[:5] + '_' + time[:2]
            rep_pic = file[-5]
            variety = var[0]
            rep_var = var[-1]
            vi = vi

            # Make dictionary for ndvi of one plot
            data = [date, time, date_time, variety, rep_var, vi, mean_m, median_m, std_m, max_m, p95_m, p90_m, p85_m,
                    rep_pic]
            nd.append(data)
        # Combine data from all images
        vi_data.extend(nd)
    # Make a Datafram to save as a csv file
    header = ['date', 'time', 'date_time', 'variety', 'rep_var', 'vi', 'mean', 'median', 'std', 'max', 'p95', 'p90',
              'p85',
              'rep_pic']
    df_final = pd.DataFrame(vi_data)
    df_final.columns = header
    df_final.to_csv(vi + '.csv')
    print('finish')

#in_path path to csv file
def image_adjustment_data(cam_name, in_path, med_arr):
    cam = 'AG9'
    month = '5'
    date = '25'

    b_med = med_arr[0]
    g_med = med_arr[1]
    r_med = med_arr[2]
    # Input path
    #in_path = 'D:/Data1/Paper_IoT/0_Raw_Data/2022/0_IoT_Image/WinterWheat/Data_Final/' + cam + '/' + month + '/' + date + '/Ref panel'

    # Read data
    df = pd.read_csv(in_path)
    # Select ref values data | max=7, mean=4
    b_ref = round(df.iloc[0, 7], 5)  # Blue
    g_ref = round(df.iloc[1, 7], 5)  # Green
    r_ref = round(df.iloc[2, 7], 5)  # Red

    # Calculate constants values
    r_b = round(b_med / b_ref, 5)
    r_g = round(g_med / g_ref, 5)
    r_r = round(r_med / r_ref, 5)

    # Add data to a dataframe
    newdf = []
    newdf['mean_ref'] = [b_med, g_med, r_med]
    newdf['constant_ref'] = [r_b, r_g, r_r]

    # save a dataframe
    df.to_csv('src\\tf2.0\\models\\research\\object_detection\\output_images\\box_1\\img_proc.csv', index=False)

# Input data

def get_image_adjustment_baseline(cam_name, in_path):
    # Input data
    cam = "AG14"
    # Input path
    #in_path = input_path_folder #'D:/Data1/Paper_IoT/0_Raw_Data/2022/0_IoT_Image/WinterWheat/Data_Final/' + cam + '/Final_Ref'  # change

    # Read data
    df = pd.read_csv(in_path, parse_dates=['date'], index_col='date').sort_index()
    # Select targeted data
    value = 'max'
    ref_val = df[['sprectrum', value]]
    r = ref_val[df['sprectrum'] == 'red'][-10:]  # Select only the last ten days data
    g = ref_val[df['sprectrum'] == 'green'][-10:]
    b = ref_val[df['sprectrum'] == 'blue'][-10:]

    # Find median value
    b_med = round(np.nanmedian(b[[value]]), 5)  # Blue
    g_med = round(np.nanmedian(g[[value]]), 5)  # Green
    r_med = round(np.nanmedian(r[[value]]), 5)  # Red

    # Make a list for ref values (row direction)
    data = [[b_med, g_med, r_med]]

    # Make a Dataframe to save as a csv file
    header = ['blue', 'green', 'red']
    df_final = pd.DataFrame(data)
    df_final.columns = header
    df_final.to_csv(in_path, index=False)
    return [b_med,g_med,r_med]


def make_constant_csv(in_path):
    r_in_path = 'src\\tf2.0\\models\\research\\object_detection\\test_images\\image_correct_demo'
    r_in_data = [img for img in os.listdir(r_in_path) if img.endswith('.png')]
    # Create an empty list to collect RGB data
    sp_data = []
    print('start')

    # Loop to extract the vi

    for file in r_in_data:
        img = cv2.imread(os.path.join(r_in_path, file))  # Read the image
        # Split the color band
        b, g, r = cv2.split(img)
        # Set list of data
        sp = [b, g, r]
        color = ['blue', 'green', 'red']
        # Create an empty list to collect data of each spectrum in the loop
        cl = []
        # For loop to extract the RGB data

        for s, c in zip(sp, color):
            index = s

            # Extract statistical data
            mean = round(np.nanmean(s), 5)
            median = round(np.nanmedian(s), 5)
            std = round(np.nanstd(s), 5)
            max = round(np.nanmax(s), 5)
            p95 = round(np.nanpercentile(s, 95), 5)
            p90 = round(np.nanpercentile(s, 90), 5)
            p85 = round(np.nanpercentile(s, 85), 5)
            date = Path(file[:].split('_')[0]).name
            time = Path(file[:].split('_')[1]).name
            date_time = date[:5] + '_' + time[:2]
            rep_pic0 = Path(file[:].split('_')[-1]).name
            rep_pic = rep_pic0.split('.')[0]
            spectrum = c
            # Make dictionary of extracted data for one color
            data = [date, time, date_time, spectrum, mean, median, std, max, p95, p90, p85, rep_pic]
            cl.append(data)

        # Combine data from all spectrum
        sp_data.extend(cl)

    # Make a Datafram to save as a csv file

    header = ['date', 'time', 'date_time', 'sprectrum', 'mean', 'median', 'std', 'max', 'p95', 'p90', 'p85', 'rep_pic']
    df_final = pd.DataFrame(sp_data)
    df_final.columns = header
    df_final.to_csv(r_in_path + '\\' + "img_data" + '.csv', index=False)
    print('finish')


make_constant_csv("")

x,y,z = get_image_adjustment_baseline("AG14",'src\\tf2.0\\models\\research\\object_detection\\test_images\\image_correct_demo\\img_data.csv')
med_arr = [x,y,z]
image_adjustment_data("AG14", "src\\tf2.0\\models\\research\\object_detection\\output_images\\box_1\\img_data.csv", med_arr)
