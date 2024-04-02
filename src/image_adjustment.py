from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import math
import random


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
    csv_length = len(df["date"])
    i=0
    # Select ref values data | max=7, mean=4
    temp_mean_ref =[]
    temp_constant_ref= []
    while (i<csv_length):
        b_ref = round(df.iloc[i+0, 7], 5)  # Blue
        g_ref = round(df.iloc[i+1, 7], 5)  # Green
        r_ref = round(df.iloc[i+2, 7], 5)  # Red

        # Calculate constants values
        r_b = round(b_med / b_ref, 5)
        r_g = round(g_med / g_ref, 5)
        r_r = round(r_med / r_ref, 5)
        i+=3
        #[b_med, g_med, r_med]
        temp_mean_ref.append(b_med)
        temp_mean_ref.append(g_med)
        temp_mean_ref.append(r_med)
        #[r_b, r_g, r_r]
        temp_constant_ref.append(r_b)
        temp_constant_ref.append(r_g)
        temp_constant_ref.append(r_r)

    # Add data to a dataframe
    df['mean_ref'] = temp_mean_ref
    df['constant_ref'] = temp_constant_ref

    # save a dataframe
    df.to_csv(in_path, index=False)



def adjust_image(r_b, r_g, r_r, img_path, plots, varieties, ndvi_data_in):
    img = cv2.imread(img_path) # Read the image
    i_w = 1280
    i_h = 1248
    img_roi = 0
    #img = img[img_roi:900, i_w:2496] # Resize the image
   
    # Split the color band
    b, g, r = cv2.split(img)
    b = r_b * b
    g = r_g * g
    r = r_r * r
    vi_data = ndvi_data_in
 
    # Calculate the ndvi
    index = ((1.664*(b.astype(float))) / (0.953*(r.astype(float)))) - 1
    # Create black image for masking
    blank = np.zeros(index.shape[:2], dtype='uint8')
    print(index)
    # cv2.imshow("i", img)
    # key = cv2.waitKey()

    nd = []

    for var, pl in zip(varieties,plots):
        # Mask the plot in left side
        # pl_m = cv2.fillPoly(blank, np.array([pl]), 255)
        # cv2.imshow("i", pl)
        # key = cv2.waitKey()
        #plots = plots[img_roi:900, i_w:2496] 
        m = cv2.bitwise_and(index, index, mask=pl)
        m[m <= 0] = np.nan  # Replace zero value to nan
        mean_m = round(np.nanmean(m), 5)
        median_m = round(np.nanmedian(m), 5)
        std_m = round(np.nanstd(m), 5)
        max_m = round(np.nanmax(m), 5)
        p95_m = round(np.nanpercentile(m, 95), 5)
        p90_m = round(np.nanpercentile(m, 90), 5)
        p85_m = round(np.nanpercentile(m, 85), 5)
        file = os.path.basename(img_path)
        date = Path(file[:].split('_')[0]).name
        time = Path(file[:].split('_')[1]).name
        date_time = date[:5] + '_' + time[:2]
        rep_pic = file[-5]
        variety = var[0]
        rep_var = var[-1]
        vi = "nvdi"

        # Make dictionary for ndvi of one plot
        data = [date, time, date_time, variety, rep_var, vi, mean_m, median_m, std_m, max_m, p95_m, p90_m, p85_m,
                rep_pic]
        nd.append(data)
    print(nd)
   
    vi_data.extend(nd)

    header = ['date', 'time', 'date_time', 'variety', 'rep_var', 'vi', 'mean', 'median', 'std', 'max', 'p95', 'p90',
              'p85',
              'rep_pic']
    df_final = pd.DataFrame(vi_data)
    df_final.columns = header

   # print(df_final)
    return df_final 
    # Mask location on the image
    # plot = [pl1, pl2, pl3]
    # var = [v1, v2, v3]
    # nd = []





def adjust_image_dummy_values(r_b, r_g, r_r, img_path, plots, varieties, ndvi_data_in):
    img = cv2.imread(img_path) # Read the image
    i_w = 1280
    i_h = 1248
    img_roi = 0
    #img = img[img_roi:900, i_w:2496] # Resize the image
   
    # Split the color band
    b, g, r = cv2.split(img)
    b = r_b * b
    g = r_g * g
    r = r_r * r
    vi_data = ndvi_data_in
 
    # Calculate the ndvi
   # index = ((1.664*(b.astype(float))) / (0.953*(r.astype(float)))) - 1
    # Create black image for masking
   # blank = np.zeros(index.shape[:2], dtype='uint8')
    #print(index)
    # cv2.imshow("i", img)
    # key = cv2.waitKey()

    nd = []

    for var, pl in zip(varieties,plots):
        # Mask the plot in left side
        # pl_m = cv2.fillPoly(blank, np.array([pl]), 255)
        # cv2.imshow("i", pl)
        # key = cv2.waitKey()
        #plots = plots[img_roi:900, i_w:2496] 
        # m = cv2.bitwise_and(index, index, mask=pl)
        # m[m <= 0] = np.nan  # Replace zero value to nan
        mean_m = 0
        median_m = 0
        std_m = 0
        max_m = 0
        p95_m = 0
        p90_m = 0
        p85_m = 0
        file = os.path.basename(img_path)
        date = Path(file[:].split('_')[0]).name
        time = Path(file[:].split('_')[1]).name
        date_time = date[:5] + '_' + time[:2]
        rep_pic = file[-5]
        variety = var[0]
        rep_var = var[-1]
        vi = "nvdi"

        # Make dictionary for ndvi of one plot
        data = [date, time, date_time, variety, rep_var, vi, mean_m, median_m, std_m, max_m, p95_m, p90_m, p85_m,
                rep_pic]
        nd.append(data)
    print(nd)
   
    vi_data.extend(nd)

    header = ['date', 'time', 'date_time', 'variety', 'rep_var', 'vi', 'mean', 'median', 'std', 'max', 'p95', 'p90',
              'p85',
              'rep_pic']
    df_final = pd.DataFrame(vi_data)
    df_final.columns = header