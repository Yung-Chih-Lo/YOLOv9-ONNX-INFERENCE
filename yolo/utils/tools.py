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

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
