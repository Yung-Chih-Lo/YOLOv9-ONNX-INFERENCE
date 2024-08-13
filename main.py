import cv2
from yolo import YOLO
import time


if __name__ == "__main__":
    image_path = 'resources/images/test.jpg'
    onnx_model_path = 'resources/weights/yolov9m-converted.onnx'
    
    image = cv2.imread(image_path)
    model = YOLO(model_path=onnx_model_path, warmup=True, device="cuda")
    start = time.perf_counter()
    results = model(image)
    print(f"inference time: {(time.perf_counter() - start)*1000:.2f} ms")
    annotated_image = model.plot()
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
