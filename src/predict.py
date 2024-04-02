#image proccessing code adapted from code by Worasit Sangjan

from ultralytics import YOLO
import cv2

import os

import numpy as np
import pandas as pd
from pathlib import Path
import math
import random









plot_model = YOLO('detect-plot.pt')


        


                








model = YOLO('detect-plate.pt')


#model.predict(source="05-06-2022_13-30-56_4.png", show=False, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=1)


#REPLACE THIS WITH YOUR LOCATION
input_images = get_input_files_list("./testset/s2") #["02-08-2022_12-00-05_1.png", "02-07-2022_10-30-04_1.png"]
#results2 = model2(input_images, conf=0.8)
results = model(input_images, conf=0.25)  # return a list of Results objects

invalid_images = []

for i in range(len(input_images)):
    result = results[i]
    image = input_images[i]
    
    box_bounding_list = result.boxes.xyxy
    
    if len(box_bounding_list) == 0:
        invalid_images.append(image)
        continue

    box_bounding_box = box_bounding_list[0]  # Assuming only one bounding box is detected
    
    corner_1 = (int(box_bounding_box[0]), int(box_bounding_box[1]))
    corner_2 = (int(box_bounding_box[2]), int(box_bounding_box[3]))

    print("before")
    slice_into_boxes(image, ".\\test_results", corner_1[0], corner_1[1],
                     corner_2[0] - corner_1[0], 4, 100, True)
    print("after")

if len(invalid_images) > 0:
    print("Could not find panel on the following images:")
    for i, invalid_image in enumerate(invalid_images):
        print(f"{invalid_image} (index: {i})")


#get data necesary for setup
make_constant_csv("./test_results")




med_arr = get_image_adjustment_baseline("test","./test_results/results.csv", f"out_baseline{i}.csv")

image_adjustment_data("test","./test_results/results.csv",med_arr)



vi_data_in = []
result = []

for i in set(input_images):
  
    plots = get_plot_mask(i, plot_model)
    # except:
    #     results = adjust_image_dummy_values(0,0,0,i,[plots[0]],["1","2","3","4","5"], vi_data_in)
    #     continue

    if i in set(invalid_images):
        results = adjust_image_dummy_values(0,0,0,i,[plots[0]],["1","2","3","4","5"], vi_data_in)
    else:
        print(i)
                
    
        b,g,r = get_r_g_b_constant_value("./test_results/results.csv",i)
        results = adjust_image(b,g,r,i,[plots[0]],["1","2","3","4","5"], vi_data_in)

pd.DataFrame(results).to_csv("./test_results/curve.csv")
    
