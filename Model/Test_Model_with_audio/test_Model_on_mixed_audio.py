import torch
import librosa
import numpy as np
import soundfile as sf
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)  
from Model.Model.model import UNet
from Model.Logging.Logger import setup_logger

Model_Test_on_audio_logger = setup_logger('ModelLogg',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI AudioEnchancer\Model\Logging\Model_logg.txt')
Model_Test_on_audio_logger.info("ModelTest on audio started...")


def load_audio(file_path, sr=44100, n_fft=2048, hop_length=512):
    #Loads the audio file and converts it to STFT
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio /= np.max(np.abs(audio)) + 1e-8
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    Model_Test_on_audio_logger.info(f"[Load_audio]: returning magnitude:  {magnitude.shape}, phase: {phase} ")
    return magnitude, phase #Returns magnitude for the model too understand the audio & phase to make sound again.



def reconstruct_audio(magnitude, phase, n_fft=2048, hop_length=512):
    stft_complex = magnitude * np.exp(1j * phase) #puts  Magntiude & phase together again
    audio = librosa.istft(stft_complex, hop_length=hop_length) #Make audio from spectrogram
    Model_Test_on_audio_logger.info(f"[reconstruct_audio]: {audio}")
    return audio #Returns the audio



def test_model(model_path, input_file, output_file, sr=44100, n_fft=2048, hop_length=512):
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    input_magnitude, input_phase = load_audio(input_file, sr, n_fft, hop_length) #Converts to spectrogram (visual representation of audio)
    Model_Test_on_audio_logger.info(f"Input magnitude shape: {input_magnitude.shape}")
    Model_Test_on_audio_logger.info(f"Input magnitude range: min={input_magnitude.min()}, max={input_magnitude.max()}")
    input_magnitude_tensor = torch.tensor(input_magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) #spectrogram to pytorch tensor that the model can understand
    Model_Test_on_audio_logger.info(f"Input device {input_magnitude_tensor.device}")
    Model_Test_on_audio_logger.info(f"model device {next(model.parameters()).device}")


    with torch.no_grad():
        output_magnitude_tensor = model(input_magnitude_tensor)
        output_magnitude = output_magnitude_tensor.squeeze().cpu().numpy()

        Model_Test_on_audio_logger.info(f"Output magnitude shape: {output_magnitude.shape}")
        Model_Test_on_audio_logger.info(f"Output magnitude range: min={output_magnitude.min()}, max={output_magnitude.max()}")

    output_audio = reconstruct_audio(output_magnitude, input_phase, n_fft, hop_length)

    sf.write(output_file, output_audio, sr)
    Model_Test_on_audio_logger.info(f"Isolerte vokaler lagret til: {output_file}")
    

if __name__ == "__main__":
    model_path = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI AudioEnchancer\UNet_Model\Final_model\Final_model.pth"
    input_file = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI AudioEnchancer\Model\Data\WAV_files_for_model\Capcut_mixed\withsound(1).WAV"
    output_file = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI AudioEnchancer\Model\Data\WAV_files_for_model\Generated_by_model.wav"


test_model(model_path,input_file,output_file)
    