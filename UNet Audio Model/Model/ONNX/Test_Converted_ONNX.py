import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

# Load the ONNX model
onnx_model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\ONNX_model\ONNX.onnx"
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

session = ort.InferenceSession(onnx_model_path)
session.set_providers(['CUDAExecutionProvider'])

# Print information about the model's input nodes
print("\n--- Model Input Nodes ---")
input_nodes = session.get_inputs()
for input_node in input_nodes:
    print(f"Name: {input_node.name}, Shape: {input_node.shape}, Type: {input_node.type}")

# Print information about the model's output nodes
print("\n--- Model Output Nodes ---")
output_nodes = session.get_outputs()
for output_node in output_nodes:
    print(f"Name: {output_node.name}, Shape: {output_node.shape}, Type: {output_node.type}")

# Show available execution providers
print("\n--- Available Providers ---")
print(f"Devices supported: {session.get_providers()}")

# Uncomment to set CUDA provider if available
#if 'CUDAExecutionProvider' in session.get_providers():
#    session.set_providers(['CUDAExecutionProvider'])

# Get input and output node names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"\nInput Name: {input_name}, Shape: {session.get_inputs()[0].shape}")
print(f"Output Name: {output_name}, Shape: {session.get_outputs()[0].shape}")

# Load an audio file and dynamically compute its spectrogram
audio_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\Model\Data\WAV_files_for_model\Capcut_mixed\withsound(1).WAV"

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found at {audio_path}")

print("\nLoading audio and generating spectrogram...")
y, sr = librosa.load(audio_path, sr=44100)  # Load audio at 44.1kHz
spectrogram = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
print(f"Original Spectrogram Shape: {spectrogram.shape}")

# Prepare input tensor
spectrogram = spectrogram.astype(np.float32)
spectrogram = np.expand_dims(spectrogram, axis=(0, 1))  # Add batch and channel dimensions
print(f"Prepared Input Tensor Shape: {spectrogram.shape}")

# Run inference
try:
    print("\nRunning ONNX inference...")
    result = session.run([output_name], {input_name: spectrogram})
    print("ONNX Inference Successful")
except Exception as e:
    print(f"Error during ONNX inference: {e}")
    raise

# Visualize the output spectrogram
output_spectrogram = np.squeeze(result[0])  # Remove batch and channel dimensions
print(f"Output Spectrogram Shape: {output_spectrogram.shape}")

plt.figure(figsize=(10, 5))
plt.imshow(output_spectrogram, aspect="auto", origin="lower")
plt.title("Vocal Isolation Output Spectrogram")
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()
