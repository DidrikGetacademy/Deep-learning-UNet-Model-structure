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
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')
import math
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

        # Collect file paths
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

        # stems shape: (stem_index, samples, channels)
        # Index 0 = mixture, index 4 = vocals (adjust based on how your data is organized)
        mixture = stems[0]
        vocals = stems[4]

        # Combine logging of shapes and sample rate
        train_logger.debug( f"[Before Mono] File: {os.path.basename(file_path)}, " f"Sample Rate: {rate}, Mixture Shape: {mixture.shape}, Vocals Shape: {vocals.shape}" )

        # Convert to mono if needed
        mixture = np.mean(mixture, axis=1) if mixture.ndim == 2 else mixture
        vocals = np.mean(vocals, axis=1) if vocals.ndim == 2 else vocals

        # Check for silent vocals
        if np.max(np.abs(vocals)) == 0:
            train_logger.info(f"Skipping file with zero-target data: {file_path}")
            return None

        # Normalize to [-1, 1] range
        mixture_max = np.max(np.abs(mixture)) + 1e-8
        vocals_max = np.max(np.abs(vocals)) + 1e-8
        mixture /= mixture_max
        vocals /= vocals_max

        train_logger.debug(f"[Normalized] mixture min={mixture.min():.4f}, max={mixture.max():.4f} | " f"vocals min={vocals.min():.4f}, max={vocals.max():.4f}")

        # Pad or trim waveforms
        mixture = self._pad_or_trim(mixture)
        vocals = self._pad_or_trim(vocals)

        train_logger.debug(f"[Padded/Trimmed] mixture length={len(mixture)}, vocals length={len(vocals)}")

        # STFT
        mixture_stft = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        vocals_stft = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)

        # Magnitudes
        mixture_mag = np.abs(mixture_stft)
        vocals_mag = np.abs(vocals_stft)

        train_logger.debug(f"[STFT Mag] mixture min={mixture_mag.min():.4f}, max={mixture_mag.max():.4f} | " f"vocals min={vocals_mag.min():.4f}, max={vocals_mag.max():.4f}")

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
        # Pad or trim the 2D spectrogram in time dimension (axis=1) to self.max_length
        time_dim = spectrogram.shape[1]
        desired_time_dim = math.ceil((self.max_length - self.n_fft) / self.hop_length) + 1
        if time_dim > desired_time_dim:
            return spectrogram[:, :desired_time_dim]
        else:
            pad_width = desired_time_dim - time_dim
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='reflect')

class VCTKSpectrogram(Dataset):
    def __init__(self, dataset, n_fft=2048, hop_length=1024, sr=44100, max_length_seconds=5, max_files=100):
        self.dataset = dataset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.max_files = max_files

        # Calculate max_length_samples based on max_length_seconds
        self.max_length_seconds = max_length_seconds
        self.max_length_samples = sr * self.max_length_seconds  # 220500 samples for 5 seconds

        # Calculate expected time dimension for STFT
        self.expected_time_dim = math.ceil((self.max_length_samples - self.n_fft) / self.hop_length) + 1  # ~215

    def __len__(self):
        return min(len(self.dataset), self.max_files)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if "audio" not in sample or "path" not in sample["audio"]:
            return None
        audio, _ = librosa.load(sample["audio"]["path"], sr=self.sr, mono=True)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Normalize
        audio = self._pad_or_trim(audio)  # Ensure consistent audio length

        # Ensure audio length is sufficient for STFT
        if len(audio) < self.n_fft:
            return None

        # Convert to STFT and compute magnitude
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=False)
        magnitude = np.abs(stft)

        # Adjust spectrogram to the expected time dimension
        magnitude = self._adjust_length(magnitude)

        # Convert to tensor and return
        return torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0), torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

    def _pad_or_trim(self, audio):
        # Ensure audio is at least n_fft in length
        if len(audio) < self.n_fft:
            padding = self.n_fft - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        # Now pad or trim to max_length_samples
        length = len(audio)
        if length < self.max_length_samples:
            padding = self.max_length_samples - length
            return np.pad(audio, (0, padding), mode='constant')
        return audio[:self.max_length_samples]

    def _adjust_length(self, spectrogram):
        # Pad or trim the 2D spectrogram in time dimension (axis=1) to self.expected_time_dim
        time_dim = spectrogram.shape[1]
        if time_dim > self.expected_time_dim:
            return spectrogram[:, :self.expected_time_dim]
        else:
            pad_width = self.expected_time_dim - time_dim
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='reflect')
