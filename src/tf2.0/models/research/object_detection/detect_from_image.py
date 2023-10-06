import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
import glob
import matplotlib.pyplot as plt
from io import BytesIO
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
            line_thickness=8)

        result_image = Image.fromarray(image_np)
        result_image.save(os.path.join(output_dir, os.path.basename(image_path)))

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
