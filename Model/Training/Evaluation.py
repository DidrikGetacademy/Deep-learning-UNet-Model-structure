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
from mir_eval.separation import bss_eval_sources
from Model.Logging.Logger import setup_logger
import csv
from joblib import Parallel, delayed
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')

#####EXTERNAL FUNCTIONS#####
def validate_and_fix_wav(file_path, output_path, sample_rate=44100):
    try:
        data, sr = sf.read(file_path)
        if sr != sample_rate or data.ndim > 1:
            data = np.mean(data, axis=1) if data.ndim > 1 else data
            sf.write(output_path, data, samplerate= sample_rate)
            train_logger.info(f"Konverterte {file_path} til {output_path} med riktig format.")
            print(f"Konverterte {file_path} til {output_path} med riktig format.")
        else: 
            print(f"{file_path} er allerede i riktig format")
    except Exception as e:
        print(f"{file_path} er allerede riktig format.")
    







#Beregner SDR, SIR og SAR ved hjelp av mir_eval.separation.bss_eval_sources.
#reference_audio (np.ndarray): shape (samples,) eller (1, samples)
#estimated_audio (np.ndarray): shape (samples,) eller (1, samples)
def compute_sdr_sir_sar(reference_audio, estimated_audio):

    if reference_audio.ndim == 1:
        reference_audio = reference_audio[np.newaxis, :]
    if estimated_audio.ndim == 1:
        estimated_audio = estimated_audio[np.newaxis, :]

    sdr, sir, sar, _ = bss_eval_sources(reference_audio, estimated_audio)
    return sdr, sir, sar










######[Eval]FUNCTION 1#######     #Klassisk MSE-baset evaluering.
def evaluate_model(model, val_loader, device, output_dir, sr=44100, n_fft=2048, hop_length=1024):

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
            output_magnitude = output_magnitude / np.max(np.abs(output_magnitude))
          
            try:
                reconstructed_audio = librosa.griffinlim(
                    output_magnitude,
                    n_iter=32,
                    n_fft=n_fft,
                    hop_length=hop_length
                )

                reconstructed_audio = reconstructed_audio / (np.max(np.abs(reconstructed_audio)) + 1e-8)
            except Exception as e:
                train_logger.error(f"Error in Griffin-Lim reconstruction (batch {idx}): {str(e)}")
                continue

   
            filename = os.path.join(output_dir, f"output_audio_{idx}.wav")
            audio_float32 = reconstructed_audio.astype(np.float32)
            sf.write(file=filename,data=audio_float32, samplerate=sr,  subtype='PCM_16')
            train_logger.info(f"Wrote audio file to {filename}")


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

        
            train_logger.info(f"Batch {idx}/{len(val_loader)} processed. "f"Loss: {loss.item():.4f}, MSE: {batch_mse:.4f}")

    if num_batches == 0:
        raise ValueError("No valid batches processed during evaluation.")

    avg_loss = total_loss / num_batches
    avg_mse = np.mean(mse_values)

    train_logger.info(f"Average loss across {num_batches} batches: {avg_loss:.4f}")
    train_logger.info(f"Average MSE across {num_batches} batches: {avg_mse:.4f}")

    return avg_loss












#####[Eval]FUNCTION 2#######
def evaluate_model_with_sdr_sir_sar( model,   dataloader,  output_csv_path,   loss_diagram_func=None,  device='cpu',  sr=44100, n_fft=2048, hop_length=512):
    model.eval()  
    sdr_list, sir_list, sar_list = [], [], []
    results = []

    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
        
        # Forbered input og target som numpy-array
        inputs, targets = inputs.numpy(), targets.numpy()

        # Modellens prediksjoner
        predictions = model(torch.tensor(inputs).to(device)).cpu().detach().numpy()

        predictions = model(torch.tensor(inputs).to(device)).cpu().detach().numpy()

        sdr_sir_sar_results = Parallel(n_jobs=1)(
            delayed(compute_sdr_sir_sar)(targets[i].flatten(), predictions[i].flatten())
            for i in range(len(targets))
        )

        # Beregn SDR, SIR og SAR for hver fil
        for i, (sdr,sir,sar) in enumerate(sdr_sir_sar_results):

            sdr_list.append(np.mean(sdr))
            sir_list.append(np.mean(sir))
            sar_list.append(np.mean(sar))

            results.append({
                "batch": batch_idx,
                "file_index": i,
                "SDR": np.mean(sdr),
                "SIR": np.mean(sir),
                "SAR": np.mean(sar),
            })


    # Logg til CSV-fil
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["batch", "file_index", "SDR", "SIR", "SAR"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Generer diagrammer hvis funksjon er gitt
    if loss_diagram_func:
        loss_diagram_func(sdr_list, sir_list, sar_list)

    # Beregn gjennomsnitt
    avg_sdr = np.mean(sdr_list)
    avg_sir = np.mean(sir_list)
    avg_sar = np.mean(sar_list)

    # Loggfør og returner
    train_logger(f"SDR: {avg_sdr}, SIR: {avg_sir}, SAR: {avg_sar}")
    return avg_sdr, avg_sir, avg_sar



