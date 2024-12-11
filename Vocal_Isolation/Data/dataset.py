import os 
import stempeg
import numpy as np
import torch 
from torch.utils.data import Dataset   
import librosa


class MUSDB18StemDataset(Dataset):
    
    def __init__(self,root_dir,subset='train',sr=44100,n_fft=2048,hop_length=512):
        
        self.root_dir = os.path.join(root_dir,subset) #Path to train/test folder
        self.sr = sr #sampling rate
        self.n_fft= n_fft #Number of FFT components
        self.hop_length = hop_length #Hop length for STFT
        self.file_paths = [os.path.join(self.root_dir,file) #Collect all `.mp4` files in the folder.
                           for file in os.listdir(self.root_dir)
                           if file.endswith('.mp4')]
        
        
        
    def __len__(self):
        return len(self.file_paths) #Returns the total number of files in the dataset. exp: in this dataset is 100
    
    
    def __getitem__(self,idx):
        file_path = self.file_paths[idx] #Get the file path for the index.
        
        stems,rate = stempeg.read_stems(file_path, sample_rate= self.sr) #Reads the (Multi-channel audio) using stempeg
        mixture = stems[0] #Mixed audio (input to the model)
        vocals = stems[4] #Isolated vocals (target output)
        
        
        #Convert stero audio to mono by averaging chhannels (if needed)
        mixture = np.mean(mixture, axis=1) if mixture.ndim == 2 else mixture
        vocals = np.mean(vocals, axis=1) if vocals.ndim == 2 else vocals
        
        
        
        #Normalize audio to a range of -1 to 1
        mixture /= np.max(np.abs(mixture))
        vocals /= np.max(np.abs(vocals))
         
         
        #Perform Short-Time fourier transform (STFT) to convert audio to spectrogram.
        mixture_stft = librosa.stft(mixture,n_fft=self.n_fft,hop_length = self.hop_length)
        vocals_stft = librosa.stft(mixture,n_fft=self.nfft,hop_length=self.hop_length)
        
        #Extract magnitude of spectrogram
        mixture_mag = np.abs(mixture_stft)
        vocals_mag = np.abs(vocals_stft)
        
        
        #Convert to pytorch tensors for training.
        mixture_mag = torch.tensor(mixture_mag,dtype=torch.float32)
        vocals_mag = torch.tensor(vocals_mag,dtype=torch.float32)
        
        return mixture_mag,vocals_mag #Returns the input and target
                