import torch
import librosa
import numpy as np
import soundfile  as sf
import matplotlib.pyplot as plt
from Model.model import UNet

#Loads a audiofile, process STFT & Returns the magnitude of spectrogram.
def load_audio(file_path,sr=44100,n_fft=1024,hop_length=256):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio /= np.max(np.abs(audio)) #Normalize to [-1, 1]
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    return magnitude, stft



#Reconstructs a audiofile from magnitude and phase with help by a invers STFT.
def reconstruct_audio(magnitude, phase, sr=44100, n_fft=1024, hop_length=256):
    stft = magnitude * np.exp(1j * phase) #Combines magnitude with phase
    audio = librosa.istft(stft, hop_length=hop_length)
    return audio




#Evaluates the model on a audiofile and stores the seperated vocals.
def evaluate_model(model_path,audio_file, output_file, sr=44100, n_fft=1024, hop_length=256):
    
    #LOAD THE MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    
    
    
    #LOAD IN & PROCESS INPUT AUDIOFILE
    input_magnitude, input_stft = load_audio(audio_file, sr ,n_fft, hop_length)
    input_phase = np.angle(input_stft) #Phase for reconstruction later
    input_tensor = torch.tensor(input_magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    
    #PERFORM PREDICATION
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_magnitude = output_tensor.squeeze().cpu().numpy()
    
    
    #RECONSTRUCT AUDIOFILE
    output_audio = reconstruct_audio(output_magnitude, input_phase, sr, n_fft, hop_length)
    
    
    #SAVE THE ISOLATED VOCAL AS AUDIOFILE
    sf.write(output_file,output_audio, sr)
    
    
    #VISUALIZE INPUT & OUTPUT SPECTROGRAM
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Input Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(input_magnitude, ref=np.max),sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\unet_vocal_isolation.pth"
    input_audio_file = r"C:\Users\didri\Desktop\AI AudioEnchancer\withinstrument.WAV"
    output_audio_file = r"C:\Users\didri\Desktop\AI AudioEnchancer\Isolated_Vocals_by_model.WAV"
    
    evaluate_model(model_path, input_audio_file, output_audio_file)