
# from ultralytics import YOLO
# import cv2

# import os

# import numpy as np
# import pandas as pd
# from pathlib import Path





# # gets the baseline values to adjust all the reference plates against
# # path to csv file
# def get_image_adjustment_baseline(cam_name, in_path):
#     # Input data
#     cam = cam_name
#     # Input path
#     #in_path = input_path_folder #'D:/Data1/Paper_IoT/0_Raw_Data/2022/0_IoT_Image/WinterWheat/Data_Final/' + cam + '/Final_Ref'  # change

#     # Read data
#     df = pd.read_csv(in_path, parse_dates=['date'], index_col='date').sort_index()
#     # Select targeted data
#     value = 'max'
#     ref_val = df[['sprectrum', value]]
#     r = ref_val[df['sprectrum'] == 'red'][-10:]  # Select only the last ten days data
#     g = ref_val[df['sprectrum'] == 'green'][-10:]
#     b = ref_val[df['sprectrum'] == 'blue'][-10:]

#     # Find median value
#     b_med = round(np.nanmedian(b[[value]]), 5)  # Blue
#     g_med = round(np.nanmedian(g[[value]]), 5)  # Green
#     r_med = round(np.nanmedian(r[[value]]), 5)  # Red

#     # Make a list for ref values (row direction)
#     data = [[b_med, g_med, r_med]]

#     # Make a Dataframe to save as a csv file
#     header = ['blue', 'green', 'red']
#     df_final = pd.DataFrame(data)
#     df_final.columns = header
#     df_final.to_csv(in_path+"out.csv", index=False)
#     return [b_med,g_med,r_med]



# # Create an empty list to collect RGB data
# def make_constant_csv(in_path):
#     r_in_path = in_path

#     r_in_path = r_in_path

#     r_in_data = [img for img in os.listdir(r_in_path) if img.endswith('.png')]

#     # Create an empty list to collect RGB data

#     sp_data = []

#     print('start')

#     # Loop to extract the vi

#     for file in r_in_data:

#         img = cv2.imread(os.path.join(r_in_path, file))  # Read the image

#         # Split the color band

#         b, g, r = cv2.split(img)

#         # Set list of data

#         sp = [b, g, r]

#         color = ['blue', 'green', 'red']

#         # Create an empty list to collect data of each spectrum in the loop

#         cl = []

#         # For loop to extract the RGB data

#         for s, c in zip(sp, color):
#             index = s

#             # Extract statistical data

#             mean = round(np.nanmean(s), 5)

#             median = round(np.nanmedian(s), 5)

#             std = round(np.nanstd(s), 5)

#             max = round(np.nanmax(s), 5)

#             p95 = round(np.nanpercentile(s, 95), 5)

#             p90 = round(np.nanpercentile(s, 90), 5)

#             p85 = round(np.nanpercentile(s, 85), 5)

#             date = Path(file[:].split('_')[0]).name

#             time = Path(file[:].split('_')[1]).name

#             date_time = date[:5] + '_' + time[:2]

#             rep_pic0 = Path(file[:].split('_')[-1]).name

#             rep_pic = rep_pic0.split('.')[0]

#             spectrum = c

#             # Make dictionary of extracted data for one color

#             data = [date, time, date_time, spectrum, mean, median, std, max, p95, p90, p85, rep_pic]

#             cl.append(data)

#         # Combine data from all spectrum

#         sp_data.extend(cl)

#     # Make a Datafram to save as a csv file

#     header = ['date', 'time', 'date_time', 'sprectrum', 'mean', 'median', 'std', 'max', 'p95', 'p90', 'p85', 'rep_pic']

#     df_final = pd.DataFrame(sp_data)

#     df_final.columns = header

#     df_final.to_csv(in_path + '\\' + "results" + '.csv', index=False)

#     print('finish')








# #last three parameter are for debug purposes only.
# def slice_into_boxes(image_path,out_folder,x,y,length,buffer=0,zoom_out=0,render_rectangle=False):
#     print(f"image path {image_path}")



#     img = cv2.imread(image_path)



#     length_base = 155
#     box_size_base = 44
#     height_base = 115
#     first_box_offset_x = int(14*(length/length_base))
#     first_box_offset_y = int(30*(length/length_base))


#     pre_out_path = out_folder+"/"+os.path.splitext(os.path.basename(image_path))[0] + "_pre_box_1" + os.path.splitext(image_path)[1]
    


#     crop_img = img.copy()
#     #cv2.imwrite(pre_out_path,crop_img)
    

#     #crop_img = img[y:int(y+(length*(height_base/length_base))), x:int(x+length)]
#     box_length =  int(length * (box_size_base / length_base))
#     cv2.rectangle(crop_img, (x, y), (int(x+length), int(y+(length*(height_base/length_base)))), (0, 0, 256), 1)
#     box_1_x_1 = int(x+first_box_offset_x+buffer)
#     box_1_y_1 = int(y+first_box_offset_y+buffer)
#     box_1_x_2 = int((x+first_box_offset_x+box_length)-(buffer))
#     box_1_y_2 = int((y+first_box_offset_y+box_length)-(buffer))

#     box_2_x_1 = int(x + first_box_offset_x+box_length+1+buffer)
#     box_2_y_1 = int(y + first_box_offset_y+buffer)
#     box_2_x_2 = int((x + first_box_offset_x + box_length*2+1-(buffer)))
#     box_2_y_2 = int(y + first_box_offset_y + box_length-(buffer))

#     box_3_x_1 = int(x + first_box_offset_x + box_length*2 + 2 +buffer)
#     box_3_y_1 = int(y + first_box_offset_y+buffer)
#     box_3_x_2 = int(x + first_box_offset_x + box_length * 3 + 2-(buffer))
#     box_3_y_2 = int(y + first_box_offset_y + box_length-(buffer))

#     crop_box_1 = crop_img[(box_1_y_1 - zoom_out):(box_1_y_2 + zoom_out), (box_1_x_1 - zoom_out):(box_1_x_2 + zoom_out)]
#     crop_box_2 = crop_img[(box_2_y_1 - zoom_out):(box_2_y_2 + zoom_out), (box_2_x_1 - zoom_out):(box_2_x_2 + zoom_out)]
#     crop_box_3 = crop_img[(box_3_y_1 - zoom_out):(box_3_y_2 + zoom_out), (box_3_x_1 - zoom_out):(box_3_x_2 + zoom_out)]
    
#     print(f"area {(box_1_x_1-box_1_x_2)*(box_1_y_2-box_1_y_1)}")
#     if render_rectangle:
#         cv2.rectangle(crop_img, (box_1_x_1,box_1_y_1),
#                       (box_1_x_2,box_1_y_2), (0,0,256), 1)

#         cv2.rectangle(crop_img, (box_2_x_1 , box_2_y_1),
#                       (box_2_x_2, box_2_y_2), (0, 256, 0), 1)
#         cv2.rectangle(crop_img, (box_3_x_1, box_3_y_1),
#                       (box_3_x_2, box_3_y_2), (256, 0, 0), 1)
#         #cv2.imwrite(os.path.splitext(image_path)[0] + "_panel" + os.path.splitext(image_path)[1], crop_img)
#     out_path = out_folder+"/"+os.path.splitext(os.path.basename(image_path))[0] + "_box_1" + os.path.splitext(image_path)[1]
#     print("out path:")
#     print(out_path)
#     cv2.imwrite(out_path,crop_box_1)
#     # cv2.imwrite(out_folder+"\\box_2\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_2" + os.path.splitext(image_path)[1], crop_box_2)
#     # cv2.imwrite(out_folder+"\\box_3\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_3" + os.path.splitext(image_path)[1], crop_box_3)





# model = YOLO("yolov8m-seg-custom.pt")

# #model.predict(source="05-06-2022_13-30-56_4.png", show=False, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=1)

# def get_input_files_list(folder):
#     dir_list = []
#     for file in os.listdir(folder):
#         if file.endswith(".png"):
#            dir_list.append( os.path.join(folder, file))
#     #print(dir_list[0:8])
#     return dir_list[0:15]


# input_images = get_input_files_list("D:\\condaEnv\\f\\wsuag-arduinoapp\\images") #["02-08-2022_12-00-05_1.png", "02-07-2022_10-30-04_1.png"]

# results = model(input_images, conf=0.8)  # return a list of Results objects

# invalid_images = []

# for i in range(0,len(input_images)):
#     result = results[i]
#     image = input_images[i]
#     print(f"IMAGE: {image}, {i} END")
#     #img = cv2.imread(image)

#     box_bounding_list = result.boxes.xyxy
#     # print(box_bounding_list)
#     # print(result.boxes.cls)

#     box_index = 0
#     #print(result.boxes.cls)
    
#     found_box = True
    
#     while int(result.boxes.cls[box_index]) != 1 :
#         box_index += 1
#         if box_index >= len(result.boxes.cls):
#             found_box = False
#             break
#     if not found_box:
#         invalid_images.append(invalid_images)
#         continue

#     box_bounding_box = box_bounding_list[box_index]

#     corner_1 = (int(box_bounding_box[0]), int(box_bounding_box[1]))
#     corner_2 = (int(box_bounding_box[2]), int(box_bounding_box[3]))
#     #img = cv2.op
#     # cv2.rectangle(img, corner_1, corner_2, color=(255,0,0), thickness=1)
#     # cv2.imwrite("./test_results/" + image, img)   
#     print("before")
#     slice_into_boxes(image,"D:\\condaEnv\\f\\wsuag-arduinoapp\\test_results",corner_1[0],corner_1[1],corner_2[0]-corner_1[0],3,100,True)
#     print("after")
# if len(invalid_images) > 0:
#     print("could not find panel on images:")
#     for i in range(0,len(invalid_images)):
#         print(invalid_images[i])

# make_constant_csv("D:\\condaEnv\\f\\wsuag-arduinoapp\\test_results")

# values_base = get_image_adjustment_baseline("test","D:\\condaEnv\\f\\wsuag-arduinoapp\\test_results\\results.csv")

# print(values_base)



# # def adjust_image(vi, varieties,r_b,r_g,r_r, in_path):
# #     # image size
# #     img_roi = 0

# #     # Create a list of data in directory
# #     in_path = in_path
# #     in_data = [img for img in os.listdir(in_path) if img.endswith('.png')]

# #     #vi = 'ndvi'

# #     # Image size
# #     i_w = 1280
# #     i_h = 1248

# #     vi_data = []

# #     print('start')
# #     # Loop to extract the vi
# #     file = os.path.basename(in_path)
# #     img = cv2.imread(in_path)  # Read the image
# #     img = img[img_roi:900, i_w:2496]  # Resize the image

# #     # Split the color band
# #     b, g, r = cv2.split(img)
# #     b = r_b * b
# #     g = r_g * g
# #     r = r_r * r

# #     # Calculate the ndvi
# #     index = ((1.664 * (b.astype(float))) / (0.953 * (r.astype(float)))) - 1
# #     # Create black image for masking
# #     blank = np.zeros(index.shape[:2], dtype='uint8')

# #     # Mask location on the image

# #     nd = []

    
# #     # Mask the plot in left side
# #     #pl_m = cv2.fillPoly(blank, np.array([pl]), 255)
# #     m = cv2.bitwise_and(index, index, mask=pl_m)
# #     m[m == 256] = np.nan  # Replace zero value to nan
# #     mean_m = round(np.nanmean(m), 5)
# #     median_m = round(np.nanmedian(m), 5)
# #     std_m = round(np.nanstd(m), 5)
# #     max_m = round(np.nanmax(m), 5)
# #     p95_m = round(np.nanpercentile(m, 95), 5)
# #     p90_m = round(np.nanpercentile(m, 90), 5)
# #     p85_m = round(np.nanpercentile(m, 85), 5)
# #     date = Path(file[:].split('_')[0]).name
# #     time = Path(file[:].split('_')[1]).name
# #     date_time = date[:5] + '_' + time[:2]
# #     rep_pic = file[-5]
# #     variety = var[0]
# #     rep_var = var[-1]
# #     vi = vi

# #         # Make dictionary for ndvi of one plot
# #     data = [date, time, date_time, variety, rep_var, vi, mean_m, median_m, std_m, max_m, p95_m, p90_m, p85_m,
# #             rep_pic]
# #     nd.append(data)
# #     # Combine data from all images
# #     vi_data.extend(nd)
# #     # Make a Datafram to save as a csv file
# #     header = ['date', 'time', 'date_time', 'variety', 'rep_var', 'vi', 'mean', 'median', 'std', 'max', 'p95', 'p90',
# #               'p85',
# #               'rep_pic']
# #     df_final = pd.DataFrame(vi_data)
# #     df_final.columns = header
# #     df_final.to_csv(vi + '.csv')
# #     print('finish')



# -------------------------------------------------------------------------------------
from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

model2 = YOLO('yolo-obb.pt')
model = YOLO('yolov8m-seg-custom.pt')

imgT = "1.png"

results = model.predict(source=imgT, conf=0.8)
results2 = model2.predict(source=imgT, conf=0.2, save = True, hide_conf=True, hide_labels=True)

for result in results:
    img = np.copy(result.orig_img)
    img_name = Path(result.path).stem  

    for contour_idx, contour in enumerate(result):

        # label = result.names[result.boxes.cls.tolist().pop()]
        label = ""
        for box in contour.boxes:
            class_id = int(box.data[0][-1])
            label = model.names[class_id]

        if label == 'box':
            x1, y1, x2, y2 = contour.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            isolated_crop = img[y1:y2, x1:x2]
        else:
            binary_mask = np.zeros(img.shape[:2], np.uint8)

            contour_xy = contour.masks.xy.pop()
            contour_xy = contour_xy.astype(np.int32)
            contour_xy = contour_xy.reshape(-1, 1, 2)

            _ = cv.drawContours(binary_mask, [contour_xy], -1, (255, 255, 255), cv.FILLED)

            mask3ch = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)
            isolated = cv.bitwise_and(mask3ch, img)

            isolated_with_transparent_bg = np.dstack([img, binary_mask])

            x1, y1, x2, y2 = contour.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            isolated_crop = isolated[y1:y2, x1:x2]

        output_filename = f"isolated_{img_name}_{contour_idx}.png"
        cv.imwrite(output_filename, isolated_crop)
        print(f"Isolated image saved: {output_filename}")


for result in results2:
    img = np.copy(result.orig_img)
    img_name = Path(result.path).stem  

    for contour_idx, contour in enumerate(result):

        # label = result.names[result.boxes.cls.tolist().pop()]
        print(contour.obb)
        label = ""
        for box in contour.obb:
            class_id = int(box.data[0][-1])
            label = model.names[class_id]

        
        x1, y1, x2, y2 = contour.obb.xyxy.cpu().numpy().squeeze().astype(np.int32)
        isolated_crop = img[y1:y2, x1:x2]

        output_filename = f"isolated_{img_name}_{contour_idx}_obb.png"
        cv.imwrite(output_filename, isolated_crop)
        print(f"Isolated image saved: {output_filename}")