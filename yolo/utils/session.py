import onnxruntime
import torch

def get_onnx_session(onnx_path):
    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                                "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)  # 初始化ONNX模型推理會話
    return session