import os
import json
import numpy as np
import skimage.draw
import skimage.io
from mrcnn.config import Config
from mrcnn import model as modellib, utils

class DamageDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        # Add your class names and IDs
        self.add_class("name", 1, "plate")
        self.add_class("name", 2, "field")

        assert subset in ["train", "val"]
        annotations_file = os.path.join(dataset_dir, "annots", subset, "annotations.json")
        
        # Load annotations from the specified file
        annotations1 = json.load(open(annotations_file))
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['name'] for s in a['regions']]
            name_dict = {"plate": 1, "field": 2}
            num_ids = [name_dict[a] for a in objects]
            image_path = os.path.join(dataset_dir, "images", a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "name",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        if info["source"] != "name":
            return super(self.__class__, self).load_mask(image_id)

        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "name":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class DamageConfig(Config):
    NAME = "name"
    NUM_CLASSES = 1 + 2
    STEPS_PER_EPOCH = 160
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.8
    WEIGHT_DECAY = 0.0001
    IMAGE_MIN_DIM = 1024
    VALIDATION_STEPS = 50
    Train_ROIs_Per_Image = 200
    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]

# Update this path to your dataset
DATASET_DIR = "kangaroo4"

# Create dataset objects
train_set = DamageDataset()
train_set.load_dataset(DATASET_DIR, "train")
train_set.prepare()

test_set = DamageDataset()
test_set.load_dataset(DATASET_DIR, "val")
test_set.prepare()

# Create configuration
config = DamageConfig()

# Create Mask R-CNN model
model = modellib.MaskRCNN(mode='training', model_dir='./', config=config)

# Load pre-trained weights
weights_path = 'mask_rcnn_coco.h5'
model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=50, layers='heads')

model_path = 'o_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
