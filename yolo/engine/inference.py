# 推理引擎的核心邏輯
from typing import List, Dict
import cv2
import numpy as np
from yolo.utils import *

class YOLO:
    def __init__(self, model_path: str, conf_thres: float = 0.7, iou_thres: float = 0.3,
                 imgsz: tuple = (640, 640), warmup: bool = True):
        """
        初始化 YOLO 類別
        :param model_path: ONNX 模型路徑
        :param conf_thres: 置信度閾值，預設為 0.6
        :param iou_thres: IoU 閾值，預設為 0.3
        :param imgsz: 輸入圖像大小，預設為 (640, 640)
        :param warmup: 是否進行預熱，預設為 True
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.annotator = Annotator(model_path)  # 初始化標註器
        self.boxes = self.scores = self.class_ids = None  # 初始化檢測結果
        self.img_height = self.img_width = None  # 初始化圖像尺寸
        self.input_names = self.input_shape = self.output_names = None  # 初始化模型輸入輸出資訊
        self.input_height, self.input_width = imgsz  # 設定輸入圖像大小
        
        self._initialize_model(model_path)  # 初始化模型
        if warmup:
            self._warmup()  # 如果需要，進行預熱

    def _warmup(self):
        """預熱處理，提高後續推理速度"""
        self.img = np.zeros((1, 1, 3), dtype=np.uint8)  # 創建一個小的空白圖像
        self.detect_objects()  # 進行一次空白檢測，預熱模型

    def __call__(self, image: np.ndarray) -> List[Dict]:
        """
        當物件被呼叫時執行物件檢測
        :param image: 輸入圖像
        :return: 檢測結果列表
        """
        self.img = image
        return self.detect_objects()

    def _initialize_model(self, path: str) -> None:
        """
        初始化模型相關參數
        :param path: 模型路徑
        """
        self.session = get_onnx_session(path)  # 獲取 ONNX 會話
        self.input_names, self.input_shape = get_input_details(self.session)  # 獲取輸入詳情
        self.output_names, _ = get_output_details(self.session)  # 獲取輸出詳情

    def detect_objects(self) -> List[Dict]:
        """
        執行物件檢測並返回結果
        :return: 檢測結果列表
        """
        input_tensor = self._preprocess()  # 預處理輸入圖像
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})  # 執行推理
        return self._process_output(outputs)  # 處理輸出結果

    def _preprocess(self) -> np.ndarray:
        """
        預處理輸入影像
        :return: 預處理後的輸入張量
        """
        self.img_height, self.img_width = self.img.shape[:2]  # 獲取原始圖像尺寸
        input_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # 轉換顏色空間
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))  # 調整到模型輸入大小
        input_img = input_img.astype(np.float32) / 255.0  # 歸一化
        return input_img.transpose(2, 0, 1)[np.newaxis, ...]  # 調整維度順序

    def _process_output(self, output: List[np.ndarray]) -> List[Dict]:
        """
        處理推理結果
        :param output: 模型輸出
        :return: 處理後的檢測結果列表
        """
        predictions = np.squeeze(output[0]).T  # 壓縮並轉置輸出
        scores = np.max(predictions[:, 4:], axis=1)  # 獲取最高置信度
        if len(scores) == 0:
            return []
        class_ids = np.argmax(predictions[:, 4:], axis=1)  # 獲取類別 ID
        boxes = self._extract_resized_boxes(predictions)  # 提取並調整檢測框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)  # 非極大值抑制
        self.boxes, self.scores, self.class_ids = xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]  # 更新檢測結果
        
        return [{"class_index": label, "confidence": score, "box": bbox.tolist()}
                for bbox, score, label in zip(self.boxes, self.scores, self.class_ids)]  # 返回檢測結果列表

    def _extract_resized_boxes(self, predictions: np.ndarray) -> np.ndarray:
        """
        提取並調整檢測框尺寸
        :param predictions: 模型預測結果
        :return: 調整後的檢測框
        """
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = predictions[:, :4]
        return boxes / input_shape * np.array([self.img_width, self.img_height, self.img_width, self.img_height])  # 將檢測框調整到原始圖像尺寸

    def plot(self) -> np.ndarray :
        """
        繪製檢測結果
        :return: 繪製了檢測結果的圖像
        """
        return self.annotator.draw_detections(self.img, self.boxes, self.scores, self.class_ids)
