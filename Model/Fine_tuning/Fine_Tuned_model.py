import os
import sys
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autocast, GradScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from Model.Logging.Logger import setup_logger
from Model.Data.dataset import MUSDB18StemDataset
from Model.Model.model import UNet


fine_tune_logger = setup_logger('Fine-Tuning',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\Fine_tune_logg.txt')
General_logger = setup_logger('General',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\General.txt')

loss_history_finetuning_epoches = {
    "l1":[],
    "mse": [],
    "spectral": [],
    "perceptual": [],
    "multiscale": [],
    "combined": [],
}

def freeze_encoder(model):
    for layer in model.encoder:
        for param in layer.parameters():
            param.requires_grad = False
    model.encoder.eval()
    fine_tune_logger.info("Encoder layers frozen for fine-tuning.")


class HybridLoss:
    def __init__(self, device):
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.device = device

        fine_tune_logger.info("Loading VGGish model...")
        self.audio_model = torch.hub.load( 'harritaylor/torchvggish', 'vggish', trust_repo=True ).to(device)
        self.audio_model.eval()
        fine_tune_logger.info("VGGish model loaded and set to eval mode.")
        General_logger.info(f"finetuning --> VGGish model loaded and set to eval mode.")

        for param in self.audio_model.parameters():
            param.requires_grad = False



    def spectral_loss(self, output, target):
        loss_value = torch.mean((output - target) ** 2)
        fine_tune_logger.debug(f"[spectral_loss] returned {loss_value.item():.6f}")
        General_logger.info(f"finetuning -->[spectral_loss] returned {loss_value.item():.6f}")
        return loss_value



    def spectrogram_batch_to_audio_list(self, spectrogram_batch, n_fft=2048, hop_length=512):
        spect_np = spectrogram_batch.detach().cpu().numpy()
        waveforms = []

        fine_tune_logger.debug(f"[spectrogram_batch_to_audio_list] spectrogram_batch.shape={spectrogram_batch.shape}")
        General_logger.debug(f"finetuning -->[spectrogram_batch_to_audio_list] spectrogram_batch.shape={spectrogram_batch.shape} ")
     
        for b in range(spect_np.shape[0]):
            single_mag = spect_np[b, 0]
            audio = librosa.griffinlim(
                single_mag,
                n_iter=32,
                hop_length=hop_length,
                win_length=n_fft
            )
            waveforms.append(torch.tensor(audio, device=self.device, dtype=torch.float32))

        return waveforms






    def perceptual_loss(self, output, target):
        output_waveforms = self.spectrogram_batch_to_audio_list(output)
        target_waveforms = self.spectrogram_batch_to_audio_list(target)

        batch_size = len(output_waveforms)
        orig_sr, new_sr = 44100, 16000
        max_length = 16000

        output_audio_list, target_audio_list = [], []

        for i in range(batch_size):
            out_np = output_waveforms[i].cpu().numpy()
            tgt_np = target_waveforms[i].cpu().numpy()

            out_16k = librosa.resample(out_np, orig_sr=orig_sr, target_sr=new_sr)
            tgt_16k = librosa.resample(tgt_np, orig_sr=orig_sr, target_sr=new_sr)

            if len(out_16k) < max_length:
                out_16k = np.pad(out_16k, (0, max_length - len(out_16k)))
            else:
                out_16k = out_16k[:max_length]

            if len(tgt_16k) < max_length:
                tgt_16k = np.pad(tgt_16k, (0, max_length - len(tgt_16k)))
            else:
                tgt_16k = tgt_16k[:max_length]

            output_audio_list.append(out_16k)
            target_audio_list.append(tgt_16k)

        output_audio_np = np.stack(output_audio_list, axis=0)
        target_audio_np = np.stack(target_audio_list, axis=0)

        # Extract features from VGGish
        output_features_list, target_features_list = [], []
        with torch.no_grad():
            for i in range(batch_size):
                out_feat = self.audio_model(output_audio_np[i], fs=new_sr)
                tgt_feat = self.audio_model(target_audio_np[i], fs=new_sr)
                output_features_list.append(out_feat)
                target_features_list.append(tgt_feat)

        output_features = torch.cat(output_features_list, dim=0)
        target_features = torch.cat(target_features_list, dim=0)
        loss_value = self.mse_loss(output_features, target_features)
        fine_tune_logger.debug(f"[perceptual_loss] returned {loss_value.item():.6f}")
        General_logger.debug(f"finetuning --> [perceptual_loss] returned {loss_value.item():.6f}")
  
        return loss_value





    def multi_scale_loss(self, output, target, scales=[1, 2, 4]):
        total_multi_scale_loss = 0.0
        for scale in scales:
            scaled_output = F.avg_pool2d(output, kernel_size=scale)
            scaled_target = F.avg_pool2d(target, kernel_size=scale)
            scale_loss = self.mse_loss(scaled_output, scaled_target)
            fine_tune_logger.debug(f"[multi_scale_loss] scale={scale}, scale_loss={scale_loss.item():.6f}")
            total_multi_scale_loss += scale_loss
        fine_tune_logger.debug(f"[multi_scale_loss] total for scales {scales} returned {total_multi_scale_loss.item():.6f}")
        General_logger.debug(f"finetuning --> [multi_scale_loss] total for scales {scales} returned {total_multi_scale_loss.item():.6f}")
           
        return total_multi_scale_loss





    def combined_loss(self, output, target):
        l1 = self.l1_loss(output, target)
        mse = self.mse_loss(output, target)
        spectral = self.spectral_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        multi_scale = self.multi_scale_loss(output, target)

        total_loss = (
            0.3 * l1 +
            0.3 * mse +
            0.1 * spectral +
            0.2 * perceptual +
            0.1 * multi_scale
        )

        fine_tune_logger.debug(f"[combined_loss] l1={l1.item():.6f}, mse={mse.item():.6f}, spectral={spectral.item():.6f}, " f"perceptual={perceptual.item():.6f}, multi_scale={multi_scale.item():.6f}, total={total_loss.item():.6f}" )
        General_logger.debug(f"finetuning --> [combined_loss] l1={l1.item():.6f}, mse={mse.item():.6f}, spectral={spectral.item():.6f}, " f"perceptual={perceptual.item():.6f}, multi_scale={multi_scale.item():.6f}, total={total_loss.item():.6f}")

        return total_loss, l1, mse, spectral, perceptual, multi_scale







def fine_tune_model(
    pretrained_model_path,
    fine_tuned_model_path,
    root_dir,
    batch_size=2,
    learning_rate=1e-5,
    fine_tune_epochs=20
):
  
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tune_logger.info(f"Fine-tuning on device: {device}")
    General_logger.info(f"finetuning --> Fine-tuning on device: {device}")
        


    dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='train',
        sr=44100,
        n_fft=2048,
        hop_length=512,
        max_length=3000,
        max_files=70
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    fine_tune_logger.info(f"Fine-tuning dataset created. #Files: {len(dataset)}, BatchSize: {batch_size}")
    General_logger.info(f"finetuning --> Fine-tuning dataset created. #Files: {len(dataset)}, BatchSize: {batch_size}")

    model = UNet(in_channels=1, out_channels=1).to(device)


    state_dict = torch.load(pretrained_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    fine_tune_logger.info(f"Pretrained model loaded from: {pretrained_model_path}")
    General_logger.info(f"finetuning -->Pretrained model loaded from: {pretrained_model_path} ")
   
    freeze_encoder(model)
    fine_tune_logger.info("Model encoder frozen.")
    General_logger.info(f"finetuning --> Model encoder frozen")

    loss_functions = HybridLoss(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=2,
        factor=0.5
    )


    scaler = GradScaler()

    print("Starting fine-tuning...")
    fine_tune_logger.info(f"Starting fine-tuning -> epochs: {fine_tune_epochs}, lr: {learning_rate}")
    General_logger.info(f"finetuning -->Starting fine-tuning -> epochs: {fine_tune_epochs}, lr: {learning_rate}")

    global loss_history_finetuning_epoches

    try:
        for epoch in range(fine_tune_epochs):
            model.train()
            running_loss = 0.0

            for batch_idx, data in enumerate(dataloader, start=1):
               
                if data is None:
                    fine_tune_logger.warning(f"Skipping batch {batch_idx} due to None data.")
                    continue

                inputs, targets = data
                if inputs is None or targets is None:
                    fine_tune_logger.warning(f"Skipping batch {batch_idx} due to None data.")
                    continue

                inputs, targets = inputs.to(device), targets.to(device)

        
                fine_tune_logger.debug(f"[Epoch {epoch+1}, Batch {batch_idx}] " ,   f"inputs.shape={inputs.shape}, inputs.min={inputs.min():.4f}, inputs.max={inputs.max():.4f} | "f"targets.shape={targets.shape}, targets.min={targets.min():.4f}, targets.max={targets.max():.4f}")

      
                optimizer.zero_grad()

                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                 
                    fine_tune_logger.debug(f"[Epoch {epoch+1}, Batch {batch_idx}] "f"outputs.shape={outputs.shape}, outputs.min={outputs.min():.4f}, outputs.max={outputs.max():.4f}")

                    total_loss, l1_val, mse_val, spectral_val, perceptual_val, multi_scale_val = \
                        loss_functions.combined_loss(outputs, targets)
                  
                    loss_history_finetuning_epoches["l1"].append(l1_val.item())
                    loss_history_finetuning_epoches["combined"].append(total_loss.item())
                    loss_history_finetuning_epoches["spectral"].append(spectral_val.item())
                    loss_history_finetuning_epoches["perceptual"].append(perceptual_val.item())
                    loss_history_finetuning_epoches["multiscale"].append(multi_scale_val.item())
                    loss_history_finetuning_epoches["perceptual"].append(perceptual_val.item())



                

              
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        
                running_loss += total_loss.item()
                fine_tune_logger.info(f"[Epoch {epoch+1}, Batch {batch_idx}] "f"L1={l1_val.item():.6f}, MSE={mse_val.item():.6f}, Spectral={spectral_val.item():.6f}, "f"Perceptual={perceptual_val.item():.6f}, MultiScale={multi_scale_val.item():.6f}, "f"TotalLoss={total_loss.item():.6f}")
                General_logger.info(f"finetuning --> [Epoch {epoch+1}, Batch {batch_idx}] "f"L1={l1_val.item():.6f}, MSE={mse_val.item():.6f}, Spectral={spectral_val.item():.6f}, "f"Perceptual={perceptual_val.item():.6f}, MultiScale={multi_scale_val.item():.6f}, "f"TotalLoss={total_loss.item():.6f}")

          
            avg_loss = running_loss / len(dataloader)
            fine_tune_logger.info( f"Epoch [{epoch+1}/{fine_tune_epochs}] -> Avg Loss: {avg_loss:.6f}")
            General_logger.info(f"finetuning -->Epoch [{epoch+1}/{fine_tune_epochs}] -> Avg Loss: {avg_loss:.6f}")

     
            current_lr = optimizer.param_groups[0]['lr']
            fine_tune_logger.info(f"Epoch [{epoch+1}] Completed. Current LR: {current_lr:e}")
            General_logger.info(f"finetuning --> Epoch [{epoch+1}] Completed. Current LR: {current_lr:e}")

     
            scheduler.step(avg_loss)

    except Exception as e:
        fine_tune_logger.error(f"Error during fine-tuning: {str(e)}")


    print("Saving fine-tuned model...")
    torch.save(model.state_dict(), fine_tuned_model_path)
    fine_tune_logger.info(f"Fine-tuned model saved to: {fine_tuned_model_path}")
    General_logger.info(f"finetuning -->Fine-tuned model saved to: {fine_tuned_model_path}")
    print("Fine-tuning complete.")
