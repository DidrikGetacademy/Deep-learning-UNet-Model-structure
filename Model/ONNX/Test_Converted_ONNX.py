import onnxruntime as ort
import numpy as np
import torch 


#Load the onnx model
onnx_model_path = "Vocal_Isolation_UNet.onnx"
session = ort.InferenceSession(onnx_model_path)

#Prepare input for onnx
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name 


input_tensor = torch.randn(1,1,256,256)
input_data = input_tensor.cpu().numpy()


result = session.run([output_name],{input_name: input_data})
print("ONNX Inference Result",np.array(result).shape)