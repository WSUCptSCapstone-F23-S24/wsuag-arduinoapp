import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path




# Create an empty list to collect RGB data
def make_constant_csv(in_path):
    r_in_path = in_path

    r_in_path = r_in_path

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

    df_final.to_csv(r_in_path + '\\' + "test4" + '.csv', index=False)

    print('finish')


def slice_into_boxes(image_path,out_folder,x,y,length):

    img = cv2.imread(image_path)
    length_base = 155
    box_size_base = 44
    height_base = 115
    first_box_offset_x = int(14*(length/length_base))
    first_box_offset_y = int(30*(length/length_base))



    crop_img = img.copy()

    buffer = 4

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

    zoom_out = 23
    crop_box_1 = crop_img[(box_1_y_1 - zoom_out):(box_1_y_2 + zoom_out), (box_1_x_1 - zoom_out):(box_1_x_2 + zoom_out)]
    crop_box_2 = crop_img[(box_2_y_1 - zoom_out):(box_2_y_2 + zoom_out), (box_2_x_1 - zoom_out):(box_2_x_2 + zoom_out)]
    crop_box_3 = crop_img[(box_3_y_1 - zoom_out):(box_3_y_2 + zoom_out), (box_3_x_1 - zoom_out):(box_3_x_2 + zoom_out)]
    render_rectangle = False
    print(f"area {(box_1_x_1-box_1_x_2)*(box_1_y_2-box_1_y_1)}")
    if render_rectangle:
        cv2.rectangle(crop_img, (box_1_x_1,box_1_y_1),
                      (box_1_x_2,box_1_y_2), (0,0,256), 1)

        cv2.rectangle(crop_img, (box_2_x_1 , box_2_y_1),
                      (box_2_x_2, box_2_y_2), (0, 256, 0), 1)
        cv2.rectangle(crop_img, (box_3_x_1, box_3_y_1),
                      (box_3_x_2, box_3_y_2), (256, 0, 0), 1)
        cv2.imwrite(os.path.splitext(image_path)[0] + "_panel" + os.path.splitext(image_path)[1], crop_img)
    print(out_folder+os.path.splitext(os.path.basename(image_path))[0] + "_box_1" + os.path.splitext(image_path)[1])

    cv2.imwrite(out_folder+"\\box_1\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_1" + os.path.splitext(image_path)[1],crop_box_1)
    cv2.imwrite(out_folder+"\\box_2\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_2" + os.path.splitext(image_path)[1], crop_box_2)
    cv2.imwrite(out_folder+"\\box_3\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_3" + os.path.splitext(image_path)[1], crop_box_3)

slice_into_boxes("C:\\Users\\code8\\Documents\\Flood_fill_421\\12-05-2022_10-30-23_2.png","C:\\Users\\code8\\Documents\\Flood_fill_421\\with_buf\\",642,228,58)
make_constant_csv("C:\\Users\\code8\\Documents\\Flood_fill_421\\with_buf\\")




def get_index(in_path,r_b,r_g,r_r,plots,varients):
    # image size
    img_roi = 0
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
        plot = plots
        var = varients
        nd = []
    
        for pl, var in zip(plot, var):
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
    header = ['date', 'time', 'date_time', 'variety', 'rep_var', 'vi', 'mean', 'median', 'std', 'max', 'p95', 'p90', 'p85',
              'rep_pic']
    df_final = pd.DataFrame(vi_data)
    df_final.columns = header
    df_final.to_csv(vi + '.csv')
    print('finish')
