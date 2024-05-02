from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import math
import random


def make_constant_csv(in_path):
    r_in_path = os.path.abspath(in_path)
    r_in_data = [img for img in os.listdir(r_in_path) if img.endswith('.png')]

    # Initialize DataFrame with specified columns
    columns = ['date', 'time', 'date_time', 'spectrum', 'mean', 'median', 'std', 'max', 'p95', 'p90', 'p85', 'rep_pic']
    df_final = pd.DataFrame(columns=columns)

    for file in r_in_data:
        img = cv2.imread(os.path.join(r_in_path, file))
        if img is None:
            print(f"Failed to load image {file}")
            continue

        b, g, r = cv2.split(img)
        channels = [b, g, r]
        colors = ['blue', 'green', 'red']

        parts = file.split('_')
        date = Path(parts[0]).stem
        time = Path(parts[1]).stem
        date_time = f"{date[:5]}_{time[:2]}"
        rep_pic = Path(parts[-1]).stem

        # Loop through each channel and directly insert data into the DataFrame
        for channel, color in zip(channels, colors):
            row = {
                'date': date,
                'time': time,
                'date_time': date_time,
                'sprectrum': color,
                'mean': np.nanmean(channel).round(5),
                'median': np.nanmedian(channel).round(5),
                'std': np.nanstd(channel).round(5),
                'max': int(np.nanmax(channel).round(5)),
                'p95': np.percentile(channel, 95).round(5),
                'p90': np.percentile(channel, 90).round(5),
                'p85': np.percentile(channel, 85).round(5),
                'rep_pic': rep_pic
                
            }
            print(int(np.nanmax(channel).round(5)))
            # Append the row directly to the DataFrame
            df_final = df_final.append(row, ignore_index=True)
            print(df_final)
    # Save the DataFrame to a CSV file
    df_final.to_csv(os.path.join(in_path, "results.csv"), index=False)
    print(f"CSV saved at {os.path.join(in_path, 'results.csv')}")


def slice_into_boxes(image_path, out_folder, x, y, length, buffer=0, zoom_out=0, render_rectangle=False):
    #print(f"image path {image_path}")



    img = cv2.imread(image_path)



    length_base = 155
    box_size_base = 44
    height_base = 115
    first_box_offset_x = int(15*(length/length_base))
    first_box_offset_y = int(30*(length/length_base))


    pre_out_path = out_folder+"/"+os.path.splitext(os.path.basename(image_path))[0] + "_pre_box_1" + os.path.splitext(image_path)[1]
    


    crop_img = img.copy()
    #cv2.imwrite(pre_out_path,crop_img)
    

    #crop_img = img[y:int(y+(length*(height_base/length_base))), x:int(x+length)]
    box_length =  int(length * (box_size_base / length_base))
    cv2.rectangle(crop_img, (x, y), (int(x+length), int(y+(length*(height_base/length_base)))), (0, 0, 256), 1)
    box_1_x_1 = int(x+first_box_offset_x+buffer)
    box_1_y_1 = int(y+first_box_offset_y+buffer)
    box_1_x_2 = int((x+first_box_offset_x+box_length)-(buffer))
    box_1_y_2 = int((y+first_box_offset_y+box_length)-(buffer))

    box_2_x_1 = int(x + first_box_offset_x+box_length+1+buffer)
    box_2_y_1 = int(y + first_box_offset_y+buffer)
    box_2_x_2 = int((x + first_box_offset_x + box_length*2+1-(buffer)))
    box_2_y_2 = int(y + first_box_offset_y + box_length-(buffer))

    box_3_x_1 = int(x + first_box_offset_x + box_length*2 + 2 +buffer)
    box_3_y_1 = int(y + first_box_offset_y+buffer)
    box_3_x_2 = int(x + first_box_offset_x + box_length * 3 + 2-(buffer))
    box_3_y_2 = int(y + first_box_offset_y + box_length-(buffer))

    crop_box_1 = crop_img[(box_1_y_1 ):(box_1_y_2), (box_1_x_1 ):(box_1_x_2 )]

    crop_box_2 = crop_img[(box_2_y_1 - zoom_out):(box_2_y_2 + zoom_out), (box_2_x_1 - zoom_out):(box_2_x_2 + zoom_out)]
    crop_box_3 = crop_img[(box_3_y_1 - zoom_out):(box_3_y_2 + zoom_out), (box_3_x_1 - zoom_out):(box_3_x_2 + zoom_out)]
    
    print(f"area {(box_1_x_1-box_1_x_2)*(box_1_y_2-box_1_y_1)}")
    if render_rectangle:
        crop_img_debug = img.copy()
        crop_box_1_debug = crop_img_debug [(box_1_y_1 - zoom_out):(box_1_y_2 + zoom_out), (box_1_x_1 - zoom_out):(box_1_x_2 + zoom_out)]
        cv2.rectangle(crop_img_debug , (box_1_x_1,box_1_y_1),
                      (box_1_x_2,box_1_y_2), (0,0,256), 1)

        cv2.rectangle(crop_img_debug , (box_2_x_1 , box_2_y_1),
                      (box_2_x_2, box_2_y_2), (0, 256, 0), 1)
        cv2.rectangle(crop_img_debug , (box_3_x_1, box_3_y_1),
                      (box_3_x_2, box_3_y_2), (256, 0, 0), 1)
        cv2.rectangle(crop_img_debug, (x, y), (int(x+length), int(y+(length*(height_base/length_base)))), (0, 0, 256), 1)
        #cv2.imwrite(os.path.splitext(image_path)[0] + "_panel" + os.path.splitext(image_path)[1], crop_img)
    out_path = out_folder+"/"+os.path.splitext(os.path.basename(image_path))[0] + "_box_1" + os.path.splitext(image_path)[1]
    # print("out path:")
    #print(out_path)
    cv2.imwrite(out_path,crop_box_1)
    if render_rectangle:
        out_path_debug = out_folder+"/debug/"+os.path.splitext(os.path.basename(image_path))[0] + "_box_1" + os.path.splitext(image_path)[1]
        #print("out path:")
        #print(out_path)
        cv2.imwrite(out_path_debug,crop_box_1_debug)

        
    # cv2.imwrite(out_folder+"\\box_2\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_2" + os.path.splitext(image_path)[1], crop_box_2)
    # cv2.imwrite(out_folder+"\\box_3\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_3" + os.path.splitext(image_path)[1], crop_box_3)

def get_input_files_list(folder):
    dir_list = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
           dir_list.append( os.path.join(folder, file))
    #print(dir_list[0:8])
    return dir_list