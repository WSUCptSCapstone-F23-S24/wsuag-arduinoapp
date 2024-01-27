from ultralytics import YOLO
import cv2

import os

import numpy as np
import pandas as pd
from pathlib import Path


def slice_into_boxes(image_path,out_folder,x,y,length,buffer=0,zoom_out=0):

    img = cv2.imread(image_path)
    length_base = 155
    box_size_base = 44
    height_base = 115
    first_box_offset_x = int(14*(length/length_base))
    first_box_offset_y = int(30*(length/length_base))



    crop_img = img.copy()

    

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

    crop_box_1 = crop_img[(box_1_y_1 - zoom_out):(box_1_y_2 + zoom_out), (box_1_x_1 - zoom_out):(box_1_x_2 + zoom_out)]
    crop_box_2 = crop_img[(box_2_y_1 - zoom_out):(box_2_y_2 + zoom_out), (box_2_x_1 - zoom_out):(box_2_x_2 + zoom_out)]
    crop_box_3 = crop_img[(box_3_y_1 - zoom_out):(box_3_y_2 + zoom_out), (box_3_x_1 - zoom_out):(box_3_x_2 + zoom_out)]
    render_rectangle = True
    print(f"area {(box_1_x_1-box_1_x_2)*(box_1_y_2-box_1_y_1)}")
    if render_rectangle:
        cv2.rectangle(crop_img, (box_1_x_1,box_1_y_1),
                      (box_1_x_2,box_1_y_2), (0,0,256), 1)

        cv2.rectangle(crop_img, (box_2_x_1 , box_2_y_1),
                      (box_2_x_2, box_2_y_2), (0, 256, 0), 1)
        cv2.rectangle(crop_img, (box_3_x_1, box_3_y_1),
                      (box_3_x_2, box_3_y_2), (256, 0, 0), 1)
        cv2.imwrite(os.path.splitext(image_path)[0] + "_panel" + os.path.splitext(image_path)[1], crop_img)
    out_path = out_folder+"/"+os.path.splitext(os.path.basename(image_path))[0] + "_box_1" + os.path.splitext(image_path)[1]
    print("out path:")
    print(out_path)
    cv2.imwrite(out_path,crop_box_1)
    # cv2.imwrite(out_folder+"\\box_2\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_2" + os.path.splitext(image_path)[1], crop_box_2)
    # cv2.imwrite(out_folder+"\\box_3\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_3" + os.path.splitext(image_path)[1], crop_box_3)





model = YOLO("yolov8m-seg-custom.pt")

#model.predict(source="05-06-2022_13-30-56_4.png", show=False, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=1)

def get_input_files_list(folder):
    dir_list = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
           dir_list.append( os.path.join(folder, file))

    print(dir_list[0:10])
    return dir_list[0:10]


input_images = get_input_files_list("C:\\Users\\code8\\Downloads\\6\\6") #["02-08-2022_12-00-05_1.png", "02-07-2022_10-30-04_1.png"]

results = model(input_images)  # return a list of Results objects



for i in range(0,len(input_images)):
    result = results[i]
    image = input_images[i]
    #img = cv2.imread(image)

    box_bounding_list = result.boxes.xyxy
    print(box_bounding_list)
    print(result.boxes.cls)

    box_index = 0
    print(result.boxes.cls)
    
    found_box = True
    
    while int(result.boxes.cls[box_index]) != 1 :
        box_index += 1
        if box_index >= len(result.boxes.cls):
            found_box = False
            break
    if not found_box:
        print(f"error could not find reference panelon image: {image}")
        continue

    box_bounding_box = box_bounding_list[box_index]

    corner_1 = (int(box_bounding_box[0]), int(box_bounding_box[1]))
    corner_2 = (int(box_bounding_box[2]), int(box_bounding_box[3]))
    #img = cv2.op
    # cv2.rectangle(img, corner_1, corner_2, color=(255,0,0), thickness=1)
    # cv2.imwrite("./test_results/" + image, img)   
    print("before")
    slice_into_boxes(image,"C:\\421_project\\wsuag-arduinoapp\\test_results",corner_1[0],corner_1[1],corner_2[0]-corner_1[0],1,5)
    print("after")


