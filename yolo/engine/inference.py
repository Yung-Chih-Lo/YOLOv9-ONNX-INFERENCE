# 推理引擎的核心邏輯
import time
import cv2
import numpy as np
from yolo.utils import xywh2xyxy, get_onnx_session, get_input_details, get_output_details
from yolo.utils import Annotator


class YOLO():
    def __init__(self, model_path:str, 
                 conf_thres:float=0.7, 
                 iou_thres:float=0.5,
                 imgsz:tuple=(640, 640), 
                 warmup:bool=True):
        self.conf_threshold = conf_thres  # 設定信心閾值
        self.iou_threshold = iou_thres  # 設定IoU（交集並集比）閾值
        self.imgsz = imgsz
        self.annotator = Annotator(model_path)
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
        
        # 初始化模型
        self.initialize_model(model_path)
        if warmup:
            self.warmup()

    def warmup(self):
        """warmup，預熱處理，後面推理比較快
        """
        self.img = np.zeros((1, 1, 3), dtype=np.uint8)
        self.detect_objects()

    def __call__(self, image):
        """
        當物件被呼叫時，執行物件檢測
        """
        self.img = image
        return self.detect_objects()  

    def initialize_model(self, path:str) -> None:
        """初始化 session, input_names, input_shape, input_height, input_width, output_names, output_shape 

        Args:
            path (str): 模型路徑
        """
        self.session = get_onnx_session(path) # 獲取onnx session
        self.input_names, self.input_shape = get_input_details(self.session) # 獲取模型輸入輸出資訊
        self.input_height, self.input_width = self.imgsz
        self.output_names, self.output_shape = get_output_details(self.session)

    def detect_objects(self) -> dict:
        """執行物件檢測

        Returns:
            list: 檢測框、分數和類別ID [boxes, scores, class_ids]
        """
        start = time.perf_counter()
        input_tensor = self.preprocess()  # 將輸入的影像進行預處理
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})  # 執行推理
        results = self.process_output(outputs)  # 處理推理結果
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return results  # 返回檢測框、分數和類別ID

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

    def process_output(self, output:list) -> dict:
        """處理推理結果

        Args:
            output (list): 推理結果

        Returns:
            list: 檢測框、分數和類別ID [boxes, scores, class_ids]
        """
        predictions = np.squeeze(output[0]).T  # 擠壓和轉置輸出
        scores = np.max(predictions[:, 4:], axis=1)  # 獲取每個檢測的信心分數
        if len(scores) == 0:
            return []  # 如果沒有合格的檢測，返回空列表
        class_ids = np.argmax(predictions[:, 4:], axis=1)  # 獲取每個檢測的類別ID
        boxes = self.extract_resized_boxes(predictions)  # 提取檢測框
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.conf_threshold, nms_threshold=self.iou_threshold) # 執行非極大值抑制
        detections = []
        self.boxes, self.scores, self.class_ids = xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
        
        for bbox, score, label in zip(self.boxes, self.scores, self.class_ids):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
            })
        return detections

    def extract_resized_boxes(self, predictions:np.ndarray) -> np.ndarray:
        """提取檢測框，並將其縮放到原始圖片尺寸

        Args:
            predictions (np.ndarray): 推理結果

        Returns:
            np.ndarray: 檢測框
        """
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = predictions[:, :4]  # 提取檢測框
        boxes = np.divide(boxes, input_shape, dtype=np.float32)  # 縮放檢測框
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])  # 調整檢測框尺寸
        return boxes

    def plot(self) -> np.ndarray:
        """繪製檢測結果

        Returns:
            np.ndarray: 繪製後的影像
        """
        return self.annotator.draw_detections(self.img, self.boxes, self.scores, self.class_ids)  # 繪製檢測結果
