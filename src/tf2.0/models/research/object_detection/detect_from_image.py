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

def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

def load_label_map(label_map_path):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def detect_objects(model, category_index, image_dir, output_dir):
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

    average_score = 0;
    image_count = 0;

    for image_path in image_paths:
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]  # Add this line to match your request
        detections = model(input_tensor)
        num_detections = int(detections['num_detections'])

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
        
        for i in range(num_detections):
            if detections['detection_scores'][0].numpy()[i] > 0.5:  # You can change the threshold as needed
                box = detections['detection_boxes'][0].numpy()[i]
                ymin, xmin, ymax, xmax = box

                # Calculate the coordinates of the top left and top right corners of the box
                x1 = int(xmin * image_np.shape[1])
                y1 = int(ymin * image_np.shape[0])
                x2 = int(xmax * image_np.shape[1])
                y2 = int(ymin * image_np.shape[0])

        print(f"Top Left: ({x1}, {y1}), Top Right: ({x2}, {y2})")

        slice_into_boxes(image_path, output_dir, x1,(y1),(x2-x1));

        # with open(os.path.join(output_dir, 'coords.txt'), 'w') as f:
        #     f.write(f'Top Left: ({x1}, {y1}), Top Right: ({x2}, {y2})')  # Access the value using mean_score[0]

        average_score +=detections['detection_scores'][0].numpy()
        image_count += 1

        box_1_output = output_dir+"\\box_1"
        process_folder(box_1_output)

        result_image = Image.fromarray(image_np)
        result_image.save(os.path.join(output_dir, os.path.basename(image_path)))

    if image_count > 0:
        mean_score = average_score / image_count
        with open(os.path.join(output_dir, 'average_score.txt'), 'w') as f:
            f.write(f'Average Detection Score: {mean_score[0]:.4f}')  # Access the value using mean_score[0]
    else:
        print("No images found in the directory.")

# Function to normalize an image using RGB values from a reference plate
def normalize_image(image_path, reference_plate_coordinates):
    # Load the image you want to normalize
    image = cv2.imread(image_path)

    # Extract the RGB values from the reference plate
    ref_plate = image[reference_plate_coordinates[0][1]:reference_plate_coordinates[1][1], 
                      reference_plate_coordinates[0][0]:reference_plate_coordinates[1][0]]
    
    # Calculate correction factors based on the reference plate values
    red_correction = ref_plate[:, :, 2].mean() / image[:, :, 2].mean()
    green_correction = ref_plate[:, :, 1].mean() / image[:, :, 1].mean()
    blue_correction = ref_plate[:, :, 0].mean() / image[:, :, 0].mean()

    # Apply the brightness correction to the entire image
    normalized_image = cv2.convertScaleAbs(image, alpha=blue_correction, beta=green_correction)

    # Determine the output file path in the same location as the original image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = os.path.dirname(image_path)
    output_file = os.path.join(output_folder, f'{image_name}_normalized.jpg')

    # Save the normalized image to the same location
    cv2.imwrite(output_file, normalized_image)

    # Optionally, display the normalized image
    cv2.imshow('Normalized Image', normalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def slice_into_boxes(image_path,out_folder,x,y,length):
    
    img = cv2.imread(image_path)
    length_base = 155
    box_size_base = 44
    height_base = 115
    first_box_offset_x = int(14*(length/length_base))
    first_box_offset_y = int(30*(length/length_base))



    crop_img = img.copy()

    buffer = 0

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

    zoom_out = 0
    crop_box_1 = crop_img[(box_1_y_1 - zoom_out):(box_1_y_2 + zoom_out), (box_1_x_1 - zoom_out):(box_1_x_2 + zoom_out)]
    crop_box_2 = crop_img[(box_2_y_1 - zoom_out):(box_2_y_2 + zoom_out), (box_2_x_1 - zoom_out):(box_2_x_2 + zoom_out)]
    crop_box_3 = crop_img[(box_3_y_1 - zoom_out):(box_3_y_2 + zoom_out), (box_3_x_1 - zoom_out):(box_3_x_2 + zoom_out)]
    render_rectangle = True
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
    cv2.imwrite(out_folder+"\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_2" + os.path.splitext(image_path)[1], crop_box_2)
    cv2.imwrite(out_folder+"\\"+os.path.splitext(os.path.basename(image_path))[0] + "_box_3" + os.path.splitext(image_path)[1], crop_box_3)
    
    

def calculate_and_update_rgb_constants(reference_plate_path, existing_constants=None):
    ref_plate = cv2.imread(reference_plate_path)

    blue_correction = ref_plate[:, :, 0].mean()
    green_correction = ref_plate[:, :, 1].mean()
    red_correction = ref_plate[:, :, 2].mean()

    if existing_constants is not None:
        # Calculate the average of existing and new constants
        blue_correction = (blue_correction + existing_constants[0]) / 2
        green_correction = (green_correction + existing_constants[1]) / 2
        red_correction = (red_correction + existing_constants[2]) / 2

    return blue_correction, green_correction, red_correction

def process_folder(folder_path):
    blue_total = 0
    green_total = 0
    red_total = 0
    num_images = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            blue, green, red = calculate_and_update_rgb_constants(image_path)
            
            # Update the running totals
            blue_total += blue
            green_total += green
            red_total += red
            num_images += 1

    # Calculate the average of RGB constants over all images
    blue_average = blue_total / num_images
    green_average = green_total / num_images
    red_average = red_total / num_images

    # Write the averages to the output file
    constants_file = os.path.join(folder_path, "constants.txt")

    # Write the averages to the output file
    with open(constants_file, 'w') as file:
        file.write(f"{blue_average} {green_average} {red_average}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection from Images')
    parser.add_argument('-m', '--model', required=True, help='Path to the model directory')
    parser.add_argument('-l', '--labelmap', required=True, help='Path to the label map')
    parser.add_argument('-i', '--images', required=True, help='Path to the input images directory')
    parser.add_argument('-o', '--output', default='output_images', help='Path to the output directory for annotated images')
    args = parser.parse_args()

    model = load_model(args.model)
    category_index = load_label_map(args.labelmap)
    os.makedirs(args.output, exist_ok=True)
    detect_objects(model, category_index, args.images, args.output)

# python .\detect_from_image.py -m .\_inference_graph\saved_model\ -l .\labelmap.pbtxt -i .\test_images\crop_test
