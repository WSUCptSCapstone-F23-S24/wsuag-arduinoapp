from ultralytics import YOLO


def initialize_plate_model(model_path='detect-plate.pt'):
    model = YOLO(model_path)
    return model


def initialize_plot_model(model_path='detect-plot.pt'):
    model = YOLO(model_path)
    return model
