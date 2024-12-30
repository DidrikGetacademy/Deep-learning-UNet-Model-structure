import os
import sys
import stempeg
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from Model.Logging.Logger import setup_logger

Dataset_logger = setup_logger('Dataset',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\Dataset_logg.txt')
General_logger = setup_logger('Dataset',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\General.txt')

Dataset_logger.info("Dataset logging started...")


class MUSDB18StemDataset(Dataset):
    def __init__(
        self,
        root_dir,
        subset='train',
        sr=44100,
        n_fft=2048,
        hop_length=512,
        max_length=10000,
        max_files=None
    ):
        self.root_dir = os.path.join(root_dir, subset)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length

        # Collect file paths
        self.file_paths = [
            os.path.join(self.root_dir, file)
            for file in os.listdir(self.root_dir)
            if file.endswith('.mp4')
        ]
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

        Dataset_logger.info(f"Initialized dataset with {len(self.file_paths)} files from '{self.root_dir}', subset='{subset}'")
     

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        Dataset_logger.info(f"Processing file: {file_path}")


        # Read stems
        try:
            stems, rate = stempeg.read_stems(file_path, sample_rate=self.sr)
        except Exception as e:
            Dataset_logger.error(f"Error reading file '{file_path}': {str(e)}")
            General_logger.error(f"Error reading file '{file_path}': {str(e)}")
            raise

        # stems shape: (stem_index, samples, channels)
        # Index 0 = mixture, index 4 = vocals (adjust based on how your data is organized)
        mixture = stems[0]
        vocals = stems[4]

        # Combine logging of shapes and sample rate
        Dataset_logger.debug( f"[Before Mono] File: {os.path.basename(file_path)}, " f"Sample Rate: {rate}, Mixture Shape: {mixture.shape}, Vocals Shape: {vocals.shape}" )
        General_logger.debug( f"[Before Mono] File: {os.path.basename(file_path)}, " f"Sample Rate: {rate}, Mixture Shape: {mixture.shape}, Vocals Shape: {vocals.shape}" )

        # Convert to mono if needed
        mixture = np.mean(mixture, axis=1) if mixture.ndim == 2 else mixture
        vocals = np.mean(vocals, axis=1) if vocals.ndim == 2 else vocals

        # Check for silent vocals
        if np.max(np.abs(vocals)) == 0:
            Dataset_logger.info(f"Skipping file with zero-target data: {file_path}")
            General_logger.info(f"Skipping file with zero-target data: {file_path}")
            return None

        # Normalize to [-1, 1] range
        mixture_max = np.max(np.abs(mixture)) + 1e-8
        vocals_max = np.max(np.abs(vocals)) + 1e-8
        mixture /= mixture_max
        vocals /= vocals_max

        Dataset_logger.debug(f"[Normalized] mixture min={mixture.min():.4f}, max={mixture.max():.4f} | " f"vocals min={vocals.min():.4f}, max={vocals.max():.4f}")
        General_logger.debug(f"[Normalized] mixture min={mixture.min():.4f}, max={mixture.max():.4f} | " f"vocals min={vocals.min():.4f}, max={vocals.max():.4f}")

        # Pad or trim waveforms
        mixture = self._pad_or_trim(mixture)
        vocals = self._pad_or_trim(vocals)

        Dataset_logger.debug(f"[Padded/Trimmed] mixture length={len(mixture)}, vocals length={len(vocals)}")
        General_logger.debug(f"[Padded/Trimmed] mixture length={len(mixture)}, vocals length={len(vocals)}")

        # STFT
        mixture_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        vocals_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)

        # Magnitudes
        mixture_mag = np.abs(mixture_stft)
        vocals_mag = np.abs(vocals_stft)

        Dataset_logger.debug(f"[STFT Mag] mixture min={mixture_mag.min():.4f}, max={mixture_mag.max():.4f} | " f"vocals min={vocals_mag.min():.4f}, max={vocals_mag.max():.4f}")
        General_logger.debug(f"[STFT Mag] mixture min={mixture_mag.min():.4f}, max={mixture_mag.max():.4f} | " f"vocals min={vocals_mag.min():.4f}, max={vocals_mag.max():.4f}")

        # Adjust spectrogram time dimension
        mixture_mag = self._adjust_length(mixture_mag)
        vocals_mag = self._adjust_length(vocals_mag)

        return (
            torch.tensor(mixture_mag, dtype=torch.float32).unsqueeze(0),
            torch.tensor(vocals_mag, dtype=torch.float32).unsqueeze(0)
        )

    def _pad_or_trim(self, audio):
        #Pad or trim the 1D audio signal to self.max_length.
  
        length = len(audio)
        if length < self.max_length:
            padding = self.max_length - length
            return np.pad(audio, (0, padding), mode='constant')
        return audio[:self.max_length]



    def _adjust_length(self, spectrogram):
        #Pad or trim the 2D spectrogram in time dimension (axis=1) to self.max_length.
       
        time_dim = spectrogram.shape[1]
        if time_dim > self.max_length:
            return spectrogram[:, :self.max_length]
        else:
            pad_width = self.max_length - time_dim
            return np.pad( spectrogram, ((0, 0), (0, pad_width)), mode='constant')
