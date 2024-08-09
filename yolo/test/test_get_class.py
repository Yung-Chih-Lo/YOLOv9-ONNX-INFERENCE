import onnx,ast

m = onnx.load('resources/weights/yolov9-c.onnx')
props = { p.key : p.value for p in m.metadata_props }
if 'names' in props:
    names = ast.literal_eval(props['names'])
    print(names)