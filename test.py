import onnx

# 加載 ONNX 模型
model = onnx.load("resources/weights/yolov9c.onnx")

# 獲取模型的輸入定義
input_tensor = model.graph.input[0]
input_tensor.type.tensor_type.shape.dim[2].dim_param =  "640" # 設置高度
input_tensor.type.tensor_type.shape.dim[3].dim_param =  "640" # 設置寬度

# 保存修改後的模型
onnx.save(model, "resources/weights/yolov9c.onnx")
