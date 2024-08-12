import onnxruntime
import torch

def get_onnx_session(onnx_path):
    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                                "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)  # 初始化ONNX模型推理會話
    return session

def get_input_details(session):
    model_inputs = session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]  # 獲取模型輸入名稱
    input_shape = model_inputs[0].shape
    input_height = input_shape[2]
    input_width = input_shape[3]
    return input_names, input_shape, input_height, input_width

def get_output_details(session):
    model_outputs = session.get_outputs()
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]  # 獲取模型輸出名稱
    output_shape = model_outputs[0].shape
    return output_names, output_shape