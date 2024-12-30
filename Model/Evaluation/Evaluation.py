import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch.nn as nn
import librosa.display
from sklearn.metrics import mean_squared_error

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from Model.Logging.Logger import setup_logger
Evaluation_logger = setup_logger('Evaluation',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\Evaluation_logg.txt')
Evaluation_logger.info("Evaluation started...")
General_logger = setup_logger('Evaluation',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\General.txt')


from mir_eval.separation import bss_eval_sources

def compute_sdr_sir_sar(reference_audio, estimated_audio):
    #Beregner SDR, SIR og SAR ved hjelp av mir_eval.separation.bss_eval_sources.
    #reference_audio (np.ndarray): shape (samples,) eller (1, samples)
    #estimated_audio (np.ndarray): shape (samples,) eller (1, samples)

    if reference_audio.ndim == 1:
        reference_audio = reference_audio[np.newaxis, :]
    if estimated_audio.ndim == 1:
        estimated_audio = estimated_audio[np.newaxis, :]

    sdr, sir, sar, perm = bss_eval_sources(reference_audio, estimated_audio)
    return sdr, sir, sar


def load_audio(file_path, sr=44100, n_fft=2048, hop_length=512):
    #Loads audio from file_path, normalizes it, returns magnitude + phase from STFT.

    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    max_val = np.max(np.abs(audio)) + 1e-8
    audio /= max_val

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    Evaluation_logger.info(f"Loaded audio: shape={audio.shape}, max={audio.max():.4f}, min={audio.min():.4f}")
    General_logger.info(f"Loaded audio: shape={audio.shape}, max={audio.max():.4f}, min={audio.min():.4f}")
    return magnitude, phase


def evaluate_model_with_sdr_sir_sar(
    model,
    val_loader,
    device,
    output_dir,
    sr=44100,
    n_fft=2048,
    hop_length=512
):


    model.eval().to(device)
    os.makedirs(output_dir, exist_ok=True)

    sdr_list, sir_list, sar_list = [], [], []
    file_count = 0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader, start=1):
       
            if batch is None:
                continue

            mixture_mag, vocals_mag = batch
            mixture_mag = mixture_mag.to(device)
            vocals_mag = vocals_mag.to(device)

            batch_size = mixture_mag.size(0)

   
            for b in range(batch_size):
                mixture_mag_b = mixture_mag[b].unsqueeze(0)  
                vocals_mag_b = vocals_mag[b]               

       
                output_vocal_mag = model(mixture_mag_b)      
                output_vocal_mag = output_vocal_mag[0, 0].cpu().numpy() 

            
                reference_vocal_mag = vocals_mag_b[0].cpu().numpy()       

         
                estimated_vocal_time = librosa.griffinlim(
                    output_vocal_mag,
                    n_iter=32,
                    n_fft=n_fft,
                    hop_length=hop_length
                )

          
                reference_vocal_time = librosa.griffinlim(
                    reference_vocal_mag,
                    n_iter=32,
                    n_fft=n_fft,
                    hop_length=hop_length
                )

           
                min_len = min(len(estimated_vocal_time), len(reference_vocal_time))
                est_wave = estimated_vocal_time[:min_len]
                ref_wave = reference_vocal_time[:min_len]

           
                est_wave = est_wave[np.newaxis, :]
                ref_wave = ref_wave[np.newaxis, :]

                sdr, sir, sar, _ = bss_eval_sources(ref_wave, est_wave)
                sdr_list.append(sdr[0])
                sir_list.append(sir[0])
                sar_list.append(sar[0])

                file_count += 1

                
                out_wav_path = os.path.join(output_dir, f"estimated_vocal_batch{idx}_sample{b+1}.wav")
                audio_float32 = est_wave[0].astype(np.float32)
                sf.write(
                    out_wav_path,
                    audio_float32,
                    sr,
                    subtype='FLOAT'  # Save as 32-bit float
                )

                Evaluation_logger.info(f"Eval idx={idx}, sample={b+1}, "f"SDR={sdr[0]:.2f}, SIR={sir[0]:.2f}, SAR={sar[0]:.2f}")
                General_logger.info(f"Eval idx={idx}, sample={b+1}, "f"SDR={sdr[0]:.2f}, SIR={sir[0]:.2f}, SAR={sar[0]:.2f}")

    if file_count == 0:
        raise ValueError("No valid samples to evaluate (perhaps all were None).")

    avg_sdr = np.mean(sdr_list)
    avg_sir = np.mean(sir_list)
    avg_sar = np.mean(sar_list)

    Evaluation_logger.info(f"Final average SDR={avg_sdr:.2f}, SIR={avg_sir:.2f}, SAR={avg_sar:.2f}")
    General_logger.info(f"Final average SDR={avg_sdr:.2f}, SIR={avg_sir:.2f}, SAR={avg_sar:.2f}")
    print(f"Evaluated {file_count} samples (across all batches) in val_loader.")
    print(f"Average SDR: {avg_sdr:.2f} dB")
    print(f"Average SIR: {avg_sir:.2f} dB")
    print(f"Average SAR: {avg_sar:.2f} dB")

    return avg_sdr, avg_sir, avg_sar





def evaluate_model(model, val_loader, device, output_dir, sr=44100, n_fft=2048, hop_length=512):
    #Klassisk MSE-baset evaluering.
    
    model.eval().to(device)
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.MSELoss()
    total_loss = 0.0
    mse_values = []
    num_batches = 0

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader, start=1):
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

 
            batch_mse = mean_squared_error(
                targets.cpu().numpy().flatten(),
                outputs.cpu().numpy().flatten()
            )
            mse_values.append(batch_mse)

        
            output_magnitude = outputs[0].cpu().numpy()

            try:
                reconstructed_audio = librosa.griffinlim(
                    output_magnitude,
                    n_iter=32,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
            except Exception as e:
                Evaluation_logger.error(f"Error in Griffin-Lim reconstruction (batch {idx}): {str(e)}")
                General_logger.error(f"Error in Griffin-Lim reconstruction (batch {idx}): {str(e)}")
                continue

   
            filename = os.path.join(output_dir, f"output_audio_{idx}.wav")
            audio_float32 = reconstructed_audio.astype(np.float32)
            sf.write(
                file=filename,
                data=audio_float32,
                samplerate=sr,
                subtype='FLOAT'  
            )
            Evaluation_logger.info(f"Wrote audio file to {filename}")
            General_logger.info(f"Wrote audio file to {filename}")

            # Plot spectrogrammer for sample 0
            input_magnitude = inputs[0].cpu().numpy()
            target_magnitude = targets[0].cpu().numpy()

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.title("Input Spectrogram")
            librosa.display.specshow(librosa.amplitude_to_db(input_magnitude.squeeze(), ref=np.max),sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
            plt.colorbar(format='%+2.0f dB')

            plt.subplot(2, 2, 2)
            plt.title("Target Spectrogram")
            librosa.display.specshow(librosa.amplitude_to_db(target_magnitude.squeeze(), ref=np.max),sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
            plt.colorbar(format='%+2.0f dB')

            plt.subplot(2, 2, 3)
            plt.title("Output Spectrogram")
            librosa.display.specshow(librosa.amplitude_to_db(output_magnitude.squeeze(), ref=np.max),sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
            plt.colorbar(format='%+2.0f dB')

            plt.subplot(2, 2, 4)
            plt.title("Reconstructed Audio Waveform")
            plt.plot(reconstructed_audio)
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")

            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"spectrograms_batch_{idx}.png")
            plt.savefig(fig_path)
            plt.close()

            Evaluation_logger.info(f"Batch {idx}/{len(val_loader)} processed. "f"Loss: {loss.item():.4f}, MSE: {batch_mse:.4f}")
            General_logger.info(f"Batch {idx}/{len(val_loader)} processed. "f"Loss: {loss.item():.4f}, MSE: {batch_mse:.4f}")

    if num_batches == 0:
        raise ValueError("No valid batches processed during evaluation.")

    avg_loss = total_loss / num_batches
    avg_mse = np.mean(mse_values)

    Evaluation_logger.info(f"Average loss across {num_batches} batches: {avg_loss:.4f}")
    General_logger.info(f"Average loss across {num_batches} batches: {avg_loss:.4f}")
    Evaluation_logger.info(f"Average MSE across {num_batches} batches: {avg_mse:.4f}")
    General_logger.info(f"Average MSE across {num_batches} batches: {avg_mse:.4f}")

    return avg_loss
