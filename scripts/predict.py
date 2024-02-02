from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8m-seg-custom.pt')

results = model.predict(source="02-06-2022_12-01-15_5.png")


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