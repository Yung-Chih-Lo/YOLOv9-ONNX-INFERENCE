import numpy as np
import onnx
import ast
import os
import cv2

# 獲取當前腳本所在的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_classes( onnx_model_path: str):
    """從 ONNX 模型中提取類別名稱"""
    names = {}
    model = onnx.load(onnx_model_path)
    props = {p.key: p.value for p in model.metadata_props}
    if 'names' in props:
        names = ast.literal_eval(props['names'])  # 字串（包含字典）轉成字典
        names = list(names.values())
    return names
    

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def preprocess(self, img: np.ndarray) -> np.ndarray:
    """將輸入的影像進行預處理，包括轉換色彩空間、調整大小、縮放像素值和調整張量維度。

        Args:
            img (np.ndarray): 輸入的影像

        Returns:
            np.ndarray: 預處理後的影像張量
    """
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (self.input_width, self.input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2,0,1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor


if __name__=="__main__":
    path="resources/weights/yolov8m.onnx"
    get_classes(path)