import onnx
onnx_model = onnx.load("TTTT.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession('TTTT.onnx')

outputs = ort_session.run(
    None,
    {'input.1': np.random.randn(1, 3, 12, 12).astype(np.float32)}
)

