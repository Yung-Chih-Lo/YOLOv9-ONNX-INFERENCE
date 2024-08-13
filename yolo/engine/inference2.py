# 推理引擎的核心邏輯
import time
import cv2
import numpy as np
import onnxruntime
import torch
from yolo.utils import xywh2xyxy, multiclass_nms, get_onnx_session, get_input_details, get_output_details
from yolo.utils import Annotator

class YOLOv9:
    def __init__(self,
                 model_path: str,
                 original_size: tuple[int, int] = (1280, 720),
                 score_thres: float = 0.1,
                 conf_thres: float = 0.4,
                 iou_thres: float = 0.4,) -> None:
        
        self.conf_threshold = conf_thres  # 設定信心閾值
        self.iou_threshold = iou_thres  # 設定IoU（交集並集比）閾值
        self.score_threshold = score_thres # 設定分數閾值, 預設0.1
        self.model_path = model_path # 模型路徑
        self.annotator = Annotator(model_path) # 這邊就會去拿到 class
        self.boxes =  None
        self.scores = None
        self.class_ids = None
        self.img_height = None
        self.img_width = None
        self.input_names = None
        self.input_shape = None
        self.input_height = None
        self.input_width = None
        self.output_names = None
        
        

        
        self.image_width, self.image_height = original_size
        self.initialize_model(model_path)

    def __call__(self, image):
        """
        當物件被呼叫時，執行物件檢測
        """
        self.img = image
        return self.detect_objects()  
    
    def detect_objects(self) -> list:
        input_tensor = self.preprocess()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return self.postprocess(outputs)
    
    def initialize_model(self, path):
        self.session = get_onnx_session(path)
        # 獲取模型輸入輸出資訊
        self.input_names, self.input_shape, self.input_height, self.input_width = get_input_details(self.session)
        self.output_names, self.output_shape = get_output_details(self.session)
                
    def preprocess(self) -> np.ndarray:
        """將輸入的影像進行預處理，包括轉換色彩空間、調整大小、縮放像素值和調整張量維度。

        Args:
            img (np.ndarray): 輸入的影像

        Returns:
            np.ndarray: 預處理後的影像張量
        """
        # TODO 考慮要不要把這個寫在utils裡面
        self.img_height, self.img_width = self.img.shape[:2]  # 獲取圖片的高度和寬度
        input_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # 將圖片轉換為RGB格式
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))  # 調整圖片大小
        input_img = input_img / 255.0  # 將像素值縮放到0到1之間
        input_img = input_img.transpose(2, 0, 1)  # 調整圖片張量的維度
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)  # 增加一個維度並轉換為float32型別
        return input_tensor

    
    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        if len(scores) == 0:
            return []  # 如果沒有合格的檢測，返回空列表
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        
        
        # Rescale box
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
        detections = []
        for bbox, score, label in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
            })
        return detections
    
        
    def plot(self) -> np.ndarray:
        """繪製檢測結果

        Returns:
            np.ndarray: 繪製後的影像
        """
        return self.annotator.draw_detections(self.img, self.boxes, self.scores, self.class_ids)  # 繪製檢測結果
    

