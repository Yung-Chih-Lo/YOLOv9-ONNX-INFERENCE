# 推理引擎的核心邏輯
import time
from typing import List, Dict
import cv2
import numpy as np
from yolo.utils import xywh2xyxy, get_onnx_session, get_input_details, get_output_details, Annotator

class YOLO:
    def __init__(self, model_path: str, conf_thres: float = 0.6, iou_thres: float = 0.4,
                 imgsz: tuple = (640, 640), warmup: bool = True):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.imgsz = imgsz
        self.annotator = Annotator(model_path)
        self.boxes = self.scores = self.class_ids = None
        self.img_height = self.img_width = None
        self.input_names = self.input_shape = self.output_names = None
        self.input_height, self.input_width = imgsz
        
        self._initialize_model(model_path)
        if warmup:
            self._warmup()

    def _warmup(self):
        """預熱處理，提高後續推理速度"""
        self.img = np.zeros((1, 1, 3), dtype=np.uint8)
        self.detect_objects()

    def __call__(self, image: np.ndarray) -> List[Dict]:
        """當物件被呼叫時執行物件檢測"""
        self.img = image
        return self.detect_objects()

    def _initialize_model(self, path: str) -> None:
        """初始化模型相關參數"""
        self.session = get_onnx_session(path)
        self.input_names, self.input_shape = get_input_details(self.session)
        self.output_names, _ = get_output_details(self.session)

    def detect_objects(self) -> List[Dict]:
        """執行物件檢測並返回結果"""
        start = time.perf_counter()
        input_tensor = self._preprocess()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        results = self._process_output(outputs)
        print(f"推理時間: {(time.perf_counter() - start)*1000:.2f} ms")
        return results

    def _preprocess(self) -> np.ndarray:
        """預處理輸入影像"""
        self.img_height, self.img_width = self.img.shape[:2]
        input_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img.astype(np.float32) / 255.0
        return input_img.transpose(2, 0, 1)[np.newaxis, ...]

    def _process_output(self, output: List[np.ndarray]) -> List[Dict]:
        """處理推理結果"""
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        if len(scores) == 0:
            return []
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self._extract_resized_boxes(predictions)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        self.boxes, self.scores, self.class_ids = xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
        
        return [{"class_index": label, "confidence": score, "box": bbox.tolist()}
                for bbox, score, label in zip(self.boxes, self.scores, self.class_ids)]

    def _extract_resized_boxes(self, predictions: np.ndarray) -> np.ndarray:
        """提取並調整檢測框尺寸"""
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = predictions[:, :4]
        return boxes / input_shape * np.array([self.img_width, self.img_height, self.img_width, self.img_height])

    def plot(self) -> np.ndarray:
        """繪製檢測結果"""
        return self.annotator.draw_detections(self.img, self.boxes, self.scores, self.class_ids)
