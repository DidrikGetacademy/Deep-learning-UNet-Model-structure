import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt




# Load the ONNX model
onnx_model_path = "Vocal_Isolation_UNet.onnx"
session = ort.InferenceSession(onnx_model_path)

#Returns information about the models input nodes
input_nodes = session.get_inputs()
for input in input_nodes:
    print(f"The models output NODES:  Name: {input.name} Shape:  {input.shape} Type: {input.type}")




#returns information about the models output nodes
output_nodes = session.get_outputs()
for output in output_nodes:
    print(f"The models output NODES:  Name: {output.name} Shape:  {output.shape} Type: {output.type}")
    

#show us what providers that is supported, CUDA,CPU ETC
print(f"Devices supported: {session.get_providers()}")

#session.set_providers(['CUDAExecutionProvider']) 

# Print input and output names for debugging
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Input Name: {input_name}, Shape: {session.get_inputs()[0].shape}")
print(f"Output Name: {output_name}, Shape: {session.get_outputs()[0].shape}")


# Load an audio file and prepare a spectrogram
audio_path = r""
y, sr = librosa.load(audio_path, sr=44100)
spectrogram = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
spectrogram = np.expand_dims(spectrogram, axis=(0, 1)).astype(np.float32)  # Add batch and channel dims

try:
    result = session.run([output_name], {input_name: spectrogram})
    print("ONNX Inference Successful")
except Exception as e:
    print(f"Error during ONNX inference: {e}")

# Visualize the output
output_spectrogram = np.squeeze(result[0])
plt.imshow(output_spectrogram, aspect="auto", origin="lower")
plt.title("Output Spectrogram")
plt.colorbar()
plt.show()
