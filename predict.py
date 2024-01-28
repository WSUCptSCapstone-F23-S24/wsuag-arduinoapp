from ultralytics import YOLO

model = YOLO("yolov8m-seg-custom.pt")

model.predict(source="03-06-2022_12-01-16_5.png", show=True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2)