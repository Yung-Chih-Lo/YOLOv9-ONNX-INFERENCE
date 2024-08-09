import cv2
import numpy as np
from .tools import get_classes

class Annotator:
    
    def __init__(self, onnx_model_path: str, mask_alpha: float = 0.3, rng_seed: int = 3,
                 font_scale_factor: float = 0.0006, text_thickness_factor: float = 0.001, 
                 box_thickness: int = 2, text_color: tuple = (255, 255, 255)):
        """
        初始化 Annotator 類。

        :param onnx_model_path: str, ONNX 模型的路徑，用於獲取類別名稱。
        :param mask_alpha: float, 遮罩的透明度。
        :param rng_seed: int, 用於生成顏色的隨機數種子。
        :param font_scale_factor: float, 字體大小比例因子。
        :param text_thickness_factor: float, 文字厚度比例因子。
        :param box_thickness: int, 繪製邊框的厚度。
        :param text_color: tuple, 繪製文字的顏色。
        """
        self.mask_alpha = mask_alpha  # mask 的 alpha 值 
        self.class_names = get_classes(onnx_model_path)
        self.font_scale_factor = font_scale_factor
        self.text_thickness_factor = text_thickness_factor
        self.box_thickness = box_thickness
        self.text_color = text_color

        # 使用隨機數生成顏色
        rng = np.random.default_rng(rng_seed)
        self.colors = rng.uniform(0, 255, size=(len(self.class_names), 3))

    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """
        在圖像上繪製檢測結果。

        :param image: np.ndarray, 原始圖像。
        :param boxes: np.ndarray, 檢測框座標。
        :param scores: np.ndarray, 檢測置信度分數。
        :param class_ids: np.ndarray, 檢測類別ID。
        :return: np.ndarray, 繪製檢測結果的圖像。
        """
        det_img = image.copy()
        img_height, img_width = image.shape[:2]

        # 計算字體大小和文字厚度
        font_size = min([img_height, img_width]) * self.font_scale_factor
        text_thickness = int(min([img_height, img_width]) * self.text_thickness_factor)

        det_img = self._draw_masks(det_img, boxes, class_ids, self.mask_alpha)

        # 繪製邊框和標籤
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = self.colors[class_id]

            self._draw_box(det_img, box, color, self.box_thickness)

            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            self._draw_text(det_img, caption, box, color, font_size, text_thickness)

        return det_img

    def _draw_box(self, image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int) -> np.ndarray:
        """
        繪製邊框。

        :param image: np.ndarray, 圖像數據。
        :param box: np.ndarray, 邊框座標。
        :param color: tuple, 邊框顏色。
        :param thickness: int, 邊框厚度。
        :return: np.ndarray, 帶有繪製邊框的圖像。
        """
        x1, y1, x2, y2 = box.astype(int)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def _draw_text(self, image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int], 
                  font_size: float, text_thickness: int) -> np.ndarray:
        """
        繪製文字標籤。

        :param image: np.ndarray, 圖像數據。
        :param text: str, 標籤文字。
        :param box: np.ndarray, 邊框座標。
        :param color: tuple, 文字背景顏色。
        :param font_size: float, 字體大小。
        :param text_thickness: int, 文字厚度。
        :return: np.ndarray, 帶有繪製文字標籤的圖像。
        """
        x1, y1, x2, y2 = box.astype(int)
        (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)

        # 繪製文字背景
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        # 繪製文字
        return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, self.text_color, text_thickness, cv2.LINE_AA)

    def _draw_masks(self, image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float) -> np.ndarray:
        """
        繪製遮罩。

        :param image: np.ndarray, 圖像數據。
        :param boxes: np.ndarray, 檢測框座標。
        :param classes: np.ndarray, 檢測類別ID。
        :param mask_alpha: float, 遮罩透明度。
        :return: np.ndarray, 帶有繪製遮罩的圖像。
        """
        mask_img = image.copy()

        # 繪製遮罩
        for box, class_id in zip(boxes, classes):
            color = self.colors[class_id]
            x1, y1, x2, y2 = box.astype(int)

            # 在遮罩圖像中繪製填充矩形
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

