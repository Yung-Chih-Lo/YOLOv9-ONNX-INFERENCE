import cv2
from yolo import YOLO


if __name__ == "__main__":
    # 加載測試圖片
    image_path = 'resources/images/test.jpg'
    onnx_model_path = 'resources/weights/yolov9c.onnx'
    
    image = cv2.imread(image_path)
    model = YOLO(onnx_model_path, warmup=True)
    results = model(image)
    annotated_image = model.plot()
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
