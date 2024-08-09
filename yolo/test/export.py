# requires onnx>=1.12.0
from ultralytics import YOLO
import json

model = YOLO("resources/weights/yolov8m.pt")
model.export(format="onnx")

with open("resources/weights/yolov8m_labels.json", "w") as labels_file:
    data = {"labels": model.names}
    json.dump(data, labels_file)