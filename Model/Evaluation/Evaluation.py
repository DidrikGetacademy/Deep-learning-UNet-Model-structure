import torch
import librosa
import numpy as np
import soundfile  as sf
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import mean_squared_error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)  
from Model.Data.dataset import MUSDB18StemDataset 
from Model.Model.model import UNet

def load_audio(file_path, sr=44100, n_fft=1024, hop_length=256):

    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio /= np.max(np.abs(audio))  # Normaliser til [-1, 1]
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    return magnitude, stft

def reconstruct_audio(magnitude, phase, sr=44100, n_fft=1024, hop_length=256):

    stft = magnitude * np.exp(1j * phase)  # Kombiner magnitude med fase
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio

def evaluate_model(model_path, audio_file, output_file,  capcut_audio=None, sr=44100, n_fft=1024, hop_length=256):

    # Last inn modellen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Last inn og prosesser input lydfil
    input_magnitude, input_stft = load_audio(audio_file, sr, n_fft, hop_length)
    input_phase = np.angle(input_stft)  # Fasen for å rekonstruere senere
    input_tensor = torch.tensor(input_magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)



    # Utfør prediksjon
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_magnitude = output_tensor.squeeze().cpu().numpy()


    # Rekonstruer lydfil
    output_audio_data = reconstruct_audio(output_magnitude, input_phase, sr, n_fft, hop_length)
    sf.write(output_file, output_audio_data, sr)
    if capcut_audio:  
        capcut_magnitude, _ = load_audio(capcut_audio, sr, n_fft, hop_length)
        mse = mean_squared_error(capcut_magnitude.flatten(), output_magnitude.flatten())
        correlation = np.corrcoef(capcut_magnitude.flatten(), output_magnitude.flatten())[0, 1]
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Correlation Coefficient: {correlation:.4f}")
        
        
        
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(input_magnitude, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 2, 2)
    plt.title("Output Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(output_magnitude, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    if capcut_audio:
        plt.figure()
        plt.title("Reference spectrogram")
        librosa.display.specshow(librosa.amplitude_to_db(capcut_magnitude, ref=np.max),sr=sr, hop_length=hop_length)
        plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# Bruk skriptet
if __name__ == "__main__":
    model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\unet_vocal_isolation.pth"
    input_audio_file = r"C:\Users\didri\Desktop\AI AudioEnchancer\Model\Data\WAV_files_for_model\Capcut_mixed\withsound(1).WAV"
    output_audio_file = r"C:\Users\didri\Desktop\AI AudioEnchancer\Model\Data\WAV_files_for_model\Generated_by_model"
    capcut_audio = r""
    evaluate_model(model_path, input_audio_file, output_audio_file)
