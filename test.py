import numpy as np
import cv2
from yolo import Annotator, YOLOv8

if __name__ == "__main__":
    # 加載測試圖片
    image_path = 'resources/images/test.jpeg'
    onnx_model_path = 'resources/weights/yolov8m.onnx'
    
    image = cv2.imread(image_path)
    model = YOLOv8(onnx_model_path)
    
    boxes, scores, class_ids = model(image)
    boxes, scores, class_ids = model(image)
    annotated_image = model.plot()
    # 顯示圖片
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()