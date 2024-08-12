# 本機測試環境
- python: 3.10.14
- cuda: 12.5
- onnx: 1.16.2
- onnxruntime: 1.18.1
- onnxruntime-gpu: 1.18.1

# 環境安裝參考（Ubuntu）

## python 
version: 3.10.14

## Visual C++ 2019 runtime
官往連結：https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

## ONNX-GPU

- 官往連結：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
- CUDA 12.x: pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
- CUDA 11.X: pip install onnxruntime-gpu

