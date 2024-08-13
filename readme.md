# Important:
- support yolov8
- support yolov9

# My environment:
- python: 3.10.14
- cuda: 12.5
- onnx: 1.16.2
- onnxruntime: 1.18.1
- onnxruntime-gpu: 1.18.1

# Installation:
```
git clone https://github.com/Yung-Chih-Lo/YOLOv9-ONNX.git
cd YOLOv9-ONNX
pip install -r requirements.txt
```
## Visual C++ 2019 runtime
- official link: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

## ONNX-GPU: 
- official link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
- CUDA 12.x: pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
- CUDA 11.X: pip install onnxruntime-gpu

## Model links:
- https://drive.google.com/drive/folders/1CZQX0LE_boLYKw5wjG3qO_1Ed59tgERm?usp=sharing


# Examples:

```
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
```

# Next step: 
- yolov10

# References:
- YOLOv9 model: https://github.com/WongKinYiu/yolov9
- YOLOv9 onnx: https://github.com/danielsyahputra/yolov9-onnx
- YOLOv8 onnx: https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection