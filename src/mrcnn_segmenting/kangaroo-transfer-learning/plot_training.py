import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class KangarooDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "box")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        all_filenames = os.listdir(images_dir)
        num_train = int(len(all_filenames) * 0.8)

        for filename in all_filenames:
            image_id = filename[:-4]

        # Use a random split based on the specified percentage
        is_train = (hash(filename) % 100) < (0.8 * 100)

        if is_train:
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

        # for filename in os.listdir(images_dir):
        #     image_id = filename[:-4]

        #     if is_train and int(image_id) >= 60:
        #         continue

        #     if not is_train and int(image_id) < 60:
        #         continue

        #     img_path = images_dir + filename
        #     ann_path = annotations_dir + image_id + '.xml'

        #     self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('box'))
        return masks, asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class KangarooConfig(mrcnn.config.Config):
    NAME = "kangaroo_cfg"
    # BACKBONE = 'efficientnetb0'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
# Train
train_dataset = KangarooDataset()
train_dataset.load_dataset(dataset_dir='kangaroo', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = KangarooDataset()
validation_dataset.load_dataset(dataset_dir='kangaroo', is_train=False)
validation_dataset.prepare()

# Model Configuration
kangaroo_config = KangarooConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=kangaroo_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=kangaroo_config.LEARNING_RATE, 
            epochs=5, 
            layers='all')

model_path = 'Kangaro_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
