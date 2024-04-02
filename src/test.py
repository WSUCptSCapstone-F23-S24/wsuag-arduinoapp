from ultralytics import YOLO

# Load a model
model = YOLO('detect-plate.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images

results = model.predict(source='testset/s2/16-07-2022_13-30-59_4.png', save=True, save_txt=True)
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
