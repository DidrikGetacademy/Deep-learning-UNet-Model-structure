import os
import sys
import stempeg
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import math
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Model.Logging.Logger import setup_logger
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')



sampling_rate = 44100
max_length_seconds = 5
max_length_samples = sampling_rate * max_length_seconds


class MUSDB18StemDataset(Dataset):
    def __init__(
        self,
        root_dir,
        subset='train',
        sr=44100,
        n_fft=2048,
        hop_length=1024,
        max_length=max_length_samples,
        max_files=100
    ):
        self.root_dir = os.path.join(root_dir, subset)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.sr = sr 


        self.file_paths = [
            os.path.join(self.root_dir, file)
            for file in os.listdir(self.root_dir)
            if file.endswith('.mp4')
        ]
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

        train_logger.info(f"Initialized dataset with {len(self.file_paths)} files from '{self.root_dir}', subset='{subset}'")
     

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]


      
        try:
            stems, rate = stempeg.read_stems(file_path, sample_rate=self.sr)
        except Exception as e:
            train_logger.error(f"Error reading file '{file_path}': {str(e)}")
            raise

     
        mixture = stems[0]
        vocals = stems[4]

    
        train_logger.debug( f"[Before Mono] File: {os.path.basename(file_path)}, " f"Sample Rate: {rate}, Mixture Shape: {mixture.shape}, Vocals Shape: {vocals.shape}" )


        mixture = np.mean(mixture, axis=1) if mixture.ndim == 2 else mixture
        vocals = np.mean(vocals, axis=1) if vocals.ndim == 2 else vocals

        if not self.has_audio(vocals) or not self.has_audio(mixture):
            print(f"Skipping file with insufficent audio {file_path}")
            return None


        if np.max(np.abs(vocals)) == 0:
            train_logger.info(f"Skipping file with zero-target data: {file_path}")
            return None

        mixture_max = np.max(np.abs(mixture)) + 1e-8
        vocals_max = np.max(np.abs(vocals)) + 1e-8
        mixture /= mixture_max
        vocals /= vocals_max

        train_logger.debug(f"[Normalized] mixture min={mixture.min():.4f}, max={mixture.max():.4f} | " f"vocals min={vocals.min():.4f}, max={vocals.max():.4f}")

    
        mixture = self._pad_or_trim(mixture)
        vocals = self._pad_or_trim(vocals)

        train_logger.debug(f"[Padded/Trimmed] mixture length={len(mixture)}, vocals length={len(vocals)}")


        mixture_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        vocals_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)

    
        mixture_mag = np.abs(mixture_stft)
        vocals_mag = np.abs(vocals_stft)

        train_logger.debug(f"[STFT Mag] mixture min={mixture_mag.min():.4f}, max={mixture_mag.max():.4f} | " f"vocals min={vocals_mag.min():.4f}, max={vocals_mag.max():.4f}")

       
        mixture_mag = self._adjust_length(mixture_mag)
        vocals_mag = self._adjust_length(vocals_mag)


        return (
            torch.tensor(mixture_mag, dtype=torch.float32).unsqueeze(0),
            torch.tensor(vocals_mag, dtype=torch.float32).unsqueeze(0)
        )
        
     #Returns true if the audio file loaded has enough valid audio for model, else it returns false.
    def has_audio(self, audio, threshold=0.01):
        if np.max(np.abs(audio)) < threshold:
            return False
        return True


    def _pad_or_trim(self, audio):
        #Pad or trim the 1D audio signal to self.max_length.
  
        length = len(audio)
        if length < self.max_length:
            padding = self.max_length - length
            return np.pad(audio, (0, padding), mode='constant')
        return audio[:self.max_length]

    



    def _adjust_length(self, spectrogram):
        # Pad or trim the 2D spectrogram in time dimension (axis=1) to self.max_length
        time_dim = spectrogram.shape[1]
        desired_time_dim = math.ceil((self.max_length - self.n_fft) / self.hop_length) + 1
        if time_dim > desired_time_dim:
            return spectrogram[:, :desired_time_dim]
        else:
            pad_width = desired_time_dim - time_dim
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='reflect')

           
#Test function for max_lengthh
def load_and_save_random_sample(self, output_dir="output"): 
        import os
        import numpy as np
        from scipy.io.wavfile import write
        import random
        os.makedirs(output_dir, exist_ok=True)

        if not self.file_paths:
            print("No files in the dataset.")
            return None
 
        random_file = random.choice(self.file_paths)

        try:
           stems, _ = stempeg.read_stems(random_file, sample_rate=self.sr)
           mixture = stems[0]  
           mixture = np.mean(mixture, axis=1) if mixture.ndim == 2 else mixture

     
           if len(mixture) > self.max_length:
               mixture = mixture[:self.max_length]
           elif len(mixture) < self.max_length:
               mixture = np.pad(mixture, (0, self.max_length - len(mixture)), mode='constant')

        
           mixture /= (np.max(np.abs(mixture)) + 1e-8)
           mixture_pcm = (mixture * 32767).astype(np.int16)

           output_path = os.path.join(output_dir, "random_max_length_sample.wav")
           write(output_path, self.sr, mixture_pcm)

           print(f"Saved random sample trimmed to max_length to: {output_path}")
           return output_path

        except Exception as e:
          print(f"Error processing file '{random_file}': {str(e)}")
          return None


