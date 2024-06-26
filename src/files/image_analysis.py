from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import math
import random


def get_image_adjustment_baseline(cam_name, in_path, csv_name="out.csv"):
    # Input data
    cam = cam_name
    # Input path
    #in_path = input_path_folder #'D:/Data1/Paper_IoT/0_Raw_Data/2022/0_IoT_Image/WinterWheat/Data_Final/' + cam + '/Final_Ref'  # change

    # Read data
    df = pd.read_csv(in_path, parse_dates=['date'], index_col='date').sort_index()
    # Select targeted data
    value = 'max'
    ref_val = df[['sprectrum', value]]
    r = ref_val[df['sprectrum'] == 'red'].sample(n=10) # Select only the last ten days data
    g = ref_val[df['sprectrum'] == 'green'].sample(n=10)
    b = ref_val[df['sprectrum'] == 'blue'].sample(n=10)

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
    df_final.to_csv(in_path+csv_name, index=False)
    return [b_med,g_med,r_med]

def get_r_g_b_constant_value(input_csv_path, input_image_path):
    file = Path(os.path.basename(input_image_path)).stem


    date = Path(file[:].split('_')[0]).name

    time1 = Path(file[:].split('_')[1]).name

    date_time = date[:5] + '_' + time1[:2]

    rep_pic0 = Path(file[:].split('_')[-1]).name

    rep_pic = rep_pic0.split('.')[0]

    df = pd.read_csv(input_csv_path)

    #print(rep_pic)
    #print(df[(df['rep_pic'] == rep_pic)])
    
    #print(df[(df['date'] == date)])
   # print( df[(df['date'] == date)&(df['time'] == time1)&(df['sprectrum'] == "red")])
    ##print("**********")
   # test = df[(df['date'] == date)&(df['time'] == time1)&(df['sprectrum'] == "red")]
    #print(test.iloc[0]["constant_ref"] )

    

    blue = df[(df['date'] == date)&(df['time'] == time1)&(df['sprectrum'] == "blue")].iloc[0]["constant_ref"]
    green = df[(df['date'] == date)&(df['time'] == time1)&(df['sprectrum'] == "green")].iloc[0]["constant_ref"]
    red = df[(df['date'] == date)&(df['time'] == time1)&(df['sprectrum'] == "red")].iloc[0]["constant_ref"]
    result = (blue,green,red)
    ##print(red)
    #print(type(red))

    return result

def get_plot_mask(img_in_path, model, expected_plot_values=[0]):
    imgT = img_in_path

    results2 = model.predict(source=imgT, conf=0.25)
    
    plot_masks = []
    plot_mask_with_expected = []
    
    
            

    for result in results2:
        img = np.copy(result.orig_img)
        img_name = Path(result.path).stem  



        for contour_idx, contour in enumerate(result):

            # label = result.names[result.boxes.cls.tolist().pop()]
            label = ""
            for box in contour.boxes:
                class_id = int(box.data[0][-1])
                label = model.names[class_id]

            if label == 'field':
                binary_mask = np.zeros(img.shape[:2], np.uint8)

                contour_xy = contour.masks.xy.pop()
                contour_xy = contour_xy.astype(np.int32)
                contour_xy = contour_xy.reshape(-1, 1, 2)

                _ = cv2.drawContours(binary_mask, [contour_xy], -1, (255, 255, 255), cv2.FILLED)

       

                mask3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                x1, y1, x2, y2 = contour.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                plot_masks.append((x1,binary_mask))

                # imS = cv2.resize(mask3ch, (960, 540))
                # cv2.imshow("i", imS)
                # key = cv2.waitKey()
                # isolated = cv2.bitwise_and(mask3ch, img)

                # isolated_with_transparent_bg = np.dstack([img, binary_mask])

                # x1, y1, x2, y2 = contour.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                # isolated_crop = isolated[y1:y2, x1:x2]

            # output_filename = f"isolated_{img_name}_{contour_idx}.png"
            # cv.imwrite(output_filename, isolated_crop)
            # print(f"Isolated image saved: {output_filename}")
    #sorts array by the x value of the top left corner 
    plot_masks = sorted(plot_masks,key=lambda x : x[0])

    plot_mask_with_expected

    plot_masks = [i[1] for i in plot_masks]



    return plot_masks