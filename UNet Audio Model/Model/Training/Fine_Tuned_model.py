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
from Model.Training.Loss_Diagram_Values import plot_loss_curves_FineTuning_script_
Fine_tune_logger = setup_logger('Fine-Tuning', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet Audio Model\Model\Logging\Model_performance_logg\Model_Training_logg.txt')


global loss_history_finetuning_epoches
num_workers = 8
loss_history_finetuning_epoches = {
    "l1":[],
    "mse": [],
    "spectral": [],
    "perceptual": [],
    "multiscale": [],
    "combined": [],
}



def Append_loss_values(loss_history_finetuning_epoches, total_loss, l1_val, multi_scale_val, perceptual_val, spectral_val, mse_val, epoch):
    #Append for batch
    loss_history_finetuning_epoches["l1"].append(l1_val.item())
    loss_history_finetuning_epoches["combined"].append(total_loss.item())
    loss_history_finetuning_epoches["spectral"].append(spectral_val.item())
    loss_history_finetuning_epoches["perceptual"].append(perceptual_val.item())
    loss_history_finetuning_epoches["multiscale"].append(multi_scale_val.item())
    loss_history_finetuning_epoches["mse"].append(mse_val.item())

    #Logging
    Fine_tune_logger.info(  f"[Train] Epoch {epoch+1}"  f"L1 Loss={l1_val.item():.6f}, mse_val={mse_val.item():.6f} , spectral loss={spectral_val.item():.6f}, perceptual_val={perceptual_val.item():.6f} ,multi_scale_val={multi_scale_val.item():.6f}, Combined Loss={total_loss.item():.6f}" )
    print(  f"[Train] Epoch {epoch+1}"  f"L1 Loss={l1_val.item():.6f}, mse_val={mse_val.item():.6f} , spectral loss={spectral_val.item():.6f}, perceptual_val={perceptual_val.item():.6f} ,multi_scale_val={multi_scale_val.item():.6f}, Combined Loss={total_loss.item():.6f}" )
 



def freeze_encoder(model):
    for layer in model.encoder:
        for param in layer.parameters():
            param.requires_grad = False
    model.encoder.eval()
    Fine_tune_logger.info("Encoder layers frozen for fine-tuning.")


class HybridLoss:
    def __init__(self, device):
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.device = device

        Fine_tune_logger.info("Loading VGGish model...")
        self.audio_model = torch.hub.load( 'harritaylor/torchvggish', 'vggish', trust_repo=True ).to(device)
        self.audio_model.eval()
        Fine_tune_logger.info(f"finetuning --> VGGish model loaded and set to eval mode.")

        for param in self.audio_model.parameters():
            param.requires_grad = False



    def spectral_loss(self, output, target):
        loss_value = torch.mean((output - target) ** 2)
        Fine_tune_logger.info(f"finetuning -->[spectral_loss] returned {loss_value.item():.6f}")
        return loss_value



    def spectrogram_batch_to_audio_list(self, spectrogram_batch, n_fft=2048, hop_length=512):
        spect_np = spectrogram_batch.detach().cpu().numpy()
        waveforms = []


        Fine_tune_logger.debug(f"finetuning -->[spectrogram_batch_to_audio_list] spectrogram_batch.shape={spectrogram_batch.shape} ")
     
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
        max_length = 176400

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
        Fine_tune_logger.debug(f"finetuning --> [perceptual_loss] returned {loss_value.item():.6f}")
  
        return loss_value





    def multi_scale_loss(self, output, target, scales=[1, 2, 4]):
        total_multi_scale_loss = 0.0
        for scale in scales:
            scaled_output = F.avg_pool2d(output, kernel_size=scale)
            scaled_target = F.avg_pool2d(target, kernel_size=scale)
            scale_loss = self.mse_loss(scaled_output, scaled_target)
            Fine_tune_logger.debug(f"[multi_scale_loss] scale={scale}, scale_loss={scale_loss.item():.6f}")
            total_multi_scale_loss += scale_loss
        Fine_tune_logger.debug(f"[multi_scale_loss] total for scales {scales} returned {total_multi_scale_loss.item():.6f}")
           
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
        Fine_tune_logger.debug(f"finetuning --> [combined_loss] l1={l1.item():.6f}, mse={mse.item():.6f}, spectral={spectral.item():.6f}, " f"perceptual={perceptual.item():.6f}, multi_scale={multi_scale.item():.6f}, total={total_loss.item():.6f}")

        return total_loss, l1, mse, spectral, perceptual, multi_scale







def fine_tune_model(
    pretrained_model_path,
    fine_tuned_model_path,
    root_dir,
    batch_size=4,
    learning_rate=1e-5,
    fine_tune_epochs=4
):
  
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fine_tune_logger.info(f"finetuning --> Fine-tuning on device: {device}")
        
   
    sampling_rate = 44100
    max_length_seconds = 5
    max_length_samples = sampling_rate * max_length_seconds

    dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='train',
        sr=44100,
        n_fft=2048,
        hop_length=1024,
        max_length=max_length_samples,
        max_files=100    
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    Fine_tune_logger.info(f"finetuning --> Fine-tuning dataset created. #Files: {len(dataset)}, BatchSize: {batch_size}")

    model = UNet(in_channels=1, out_channels=1).to(device)


    state_dict = torch.load(pretrained_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    Fine_tune_logger.info(f"finetuning -->Pretrained model loaded from: {pretrained_model_path} ")
   


    freeze_encoder(model)
    Fine_tune_logger.info(f"finetuning --> Model encoder frozen")

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
    Fine_tune_logger.info(f"finetuning -->Starting fine-tuning -> epochs: {fine_tune_epochs}, lr: {learning_rate}")

    global loss_history_finetuning_epoches

    try:
        for epoch in range(fine_tune_epochs):
            model.train()
            running_loss = 0.0

            for batch_idx, data in enumerate(dataloader, start=1):
               
                if data is None:
                    Fine_tune_logger.warning(f"Skipping batch {batch_idx} due to None data.")
                    print(f"Skipping batch {batch_idx} due to None data.")
                    continue

                inputs, targets = data
                if inputs is None or targets is None:
                    Fine_tune_logger.warning(f"Skipping batch {batch_idx} due to None data.")
                    print(f"Skipping batch {batch_idx} due to None data.")
                    continue

                inputs, targets = inputs.to(device), targets.to(device)

        
                Fine_tune_logger.debug(f"[Epoch {epoch+1}, Batch {batch_idx}] " ,   f"inputs.shape={inputs.shape}, inputs.min={inputs.min():.4f}, inputs.max={inputs.max():.4f} | "f"targets.shape={targets.shape}, targets.min={targets.min():.4f}, targets.max={targets.max():.4f}")
                print(f"[Epoch {epoch+1}, Batch {batch_idx}] " ,   f"inputs.shape={inputs.shape}, inputs.min={inputs.min():.4f}, inputs.max={inputs.max():.4f} | "f"targets.shape={targets.shape}, targets.min={targets.min():.4f}, targets.max={targets.max():.4f}")

      
                optimizer.zero_grad()

                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                 
                    Fine_tune_logger.debug(f"[Epoch {epoch+1}, Batch {batch_idx}] "f"outputs.shape={outputs.shape}, outputs.min={outputs.min():.4f}, outputs.max={outputs.max():.4f}")

                    total_loss, l1_val, mse_val, spectral_val, perceptual_val, multi_scale_val = \
                        loss_functions.combined_loss(outputs, targets)
                  


            
                Append_loss_values(loss_history_finetuning_epoches, total_loss, l1_val, multi_scale_val, perceptual_val, spectral_val, mse_val, epoch)
            
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        
                running_loss += total_loss.item()
                Fine_tune_logger.info(f"finetuning --> [Epoch {epoch+1}, Batch {batch_idx}] "f"L1={l1_val.item():.6f}, MSE={mse_val.item():.6f}, Spectral={spectral_val.item():.6f}, "f"Perceptual={perceptual_val.item():.6f}, MultiScale={multi_scale_val.item():.6f}, "f"TotalLoss={total_loss.item():.6f}")

          
            avg_loss = running_loss / len(dataloader)

            Fine_tune_logger.info(f"finetuning -->Epoch [{epoch+1}/{fine_tune_epochs}] -> Avg Loss: {avg_loss:.6f}")

     
            current_lr = optimizer.param_groups[0]['lr']
            Fine_tune_logger.info(f"finetuning --> Epoch [{epoch+1}] Completed. Current LR: {current_lr:e}")
            plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches,'loss_curves_finetuning_epoches.png')
     
            scheduler.step(avg_loss)

    except Exception as e:
        Fine_tune_logger.error(f"Error during fine-tuning: {str(e)}")



    torch.save(model.state_dict(), fine_tuned_model_path)
    Fine_tune_logger.info(f"finetuning -->Fine-tuned model saved to: {fine_tuned_model_path}")

    if __name__ == "__main__":
       fine_tune_model(pretrained_model_path=r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet Audio Model\UNet_Model_Weights\Final_model\Final_model.pth",fine_tuned_model_path=r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet Audio Model\UNet_Model_Weights\Fine_tuned_model", root_dir = r'C:\mappe1\musdb18')

   