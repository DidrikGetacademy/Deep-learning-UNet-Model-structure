import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Vocal_Isolation_Model.Model.model import UNet

def load_audio(file_path, sr=44100, n_fft=1024, hop_length=256):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio /= np.max(np.abs(audio))  # Normalize to [-1, 1]
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    return magnitude, stft

def reconstruct_audio(magnitude, phase, sr=44100, n_fft=1024, hop_length=256):
    stft = magnitude * np.exp(1j * phase)  # Combine magnitude with phase
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio

def evaluate_and_compare(model_path, input_audio, capcut_audio, output_audio, sr=44100, n_fft=1024, hop_length=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    input_magnitude, input_stft = load_audio(input_audio, sr, n_fft, hop_length)
    input_phase = np.angle(input_stft)
    input_tensor = torch.tensor(input_magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_magnitude = output_tensor.squeeze().cpu().numpy()
    
    capcut_magnitude, _ = load_audio(capcut_audio, sr, n_fft, hop_length)
    
    mse = mean_squared_error(capcut_magnitude.flatten(), output_magnitude.flatten())
    correlation = np.corrcoef(capcut_magnitude.flatten(), output_magnitude.flatten())[0, 1]
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}")
    
    output_audio_data = reconstruct_audio(output_magnitude, input_phase, sr, n_fft, hop_length)
    sf.write(output_audio, output_audio_data, sr)
    
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(input_magnitude, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(1, 3, 2)
    plt.title("Capcut Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(capcut_magnitude, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(1, 3, 3)
    plt.title("Model Output Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(output_magnitude, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\unet_vocal_isolation.pth"
    input_audio_file = r"C:\Users\didri\Desktop\AI AudioEnchancer\WAV_files\Mixed_Audio\withinstrument.WAV"
    capcut_audio = r"C:\Users\didri\Desktop\AI AudioEnchancer\WAV_files\Isolated_audio\withoutinstrument.WAV"
    output_audio_file = r"C:\Users\didri\Desktop\AI AudioEnchancer\WAV_files\Isolated_Vocals_by_model3.WAV"
    
    evaluate_and_compare(model_path, input_audio_file, capcut_audio, output_audio_file)
