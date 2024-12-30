import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import gc
from torch import autocast, GradScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from Model.Logging.Logger import setup_logger
from Model.Data.dataset import MUSDB18StemDataset
from Model.Model.model import UNet
from Model.Evaluation.Evaluation import evaluate_model, evaluate_model_with_sdr_sir_sar
from Model.Fine_tuning.Fine_Tuned_model import fine_tune_model

# Viktig: Oppdatert import av plot-funksjoner
from Model.Logging.Loss_Diagram_Values import (
    plot_loss_curves_Training_script_epoches,
    plot_loss_curves_Training_script_Batches
)

Evaluation_logger = setup_logger('Evaluation',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\Evaluation_logg.txt')
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\Train_Logg.txt')
General_logger = setup_logger('train',r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Script_logs\General.txt')



loss_history_Epoches = {
    "l1": [],
    "spectral": [],
    "combined": [],
    "Runningloss": [],
}

loss_history_Batches = {
    "l1": [],
    "spectral": [],
    "mse": [],
    "combined": [],
}

avg_trainloss = [],

class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()



    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

    
        l1 = self.l1_loss(pred, target)


        n_fft = min(2048, pred.size(-1))
        hop_length = 512
        window = torch.hann_window(n_fft, device=pred.device)

        batch_size = pred.size(0)
        stft_loss = 0.0
        for i in range(batch_size):
            pred_tensor = pred[i].squeeze()
            target_tensor = target[i].squeeze()

            min_len = min(pred_tensor.size(-1), target_tensor.size(-1))
            pred_tensor = pred_tensor[..., :min_len]
            target_tensor = target_tensor[..., :min_len]

            pred_stft = torch.stft(
                pred_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                window=window,
            )
            target_stft = torch.stft(
                target_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                window=window,
            )

            stft_loss_batch = self.mse_loss(
                torch.abs(pred_stft),
                torch.abs(target_stft)
            )
            stft_loss += stft_loss_batch

      
            train_logger.debug( f"[HybridLoss] Batch {i}: STFT MSE={stft_loss_batch.item():.6f}")
            General_logger.debug( f"[HybridLoss] Batch {i}: STFT MSE={stft_loss_batch.item():.6f}")
            

        stft_loss /= batch_size
        scaled_stft_loss = stft_loss / 1000.0
        combined_loss = 0.3 * l1 + 0.7 * scaled_stft_loss

        train_logger.debug( f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f} "f"(scaled={scaled_stft_loss.item():.6f}), Combined={combined_loss.item():.6f}")
        General_logger.debug( f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f} "f"(scaled={scaled_stft_loss.item():.6f}), Combined={combined_loss.item():.6f}")

       
        return combined_loss, l1, stft_loss


def train(load_model_path=r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\Final_model\Final_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    print(f"Device: {device}")


    batch_size = 2
    learning_rate = 1e-5
    epochs = 10
    root_dir = r'C:\mappe1\musdb18'


    diagramdirectory = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Diagram_Resultater"
    os.makedirs(diagramdirectory, exist_ok=True)

 
    output_dir = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Evaluation"
    os.makedirs(output_dir, exist_ok=True)

    
    dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='train',
        sr=44100,
        n_fft=2048,
        hop_length=512,
        max_length=3000,
        max_files=100,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='test',
        sr=44100,
        n_fft=2048,
        hop_length=512,
        max_length=3000,
        max_files=60,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


    train_logger.info(  f"[Train] Dataset => max_length: {dataset.max_length}, sr: {dataset.sr}, "f"n_fft: {dataset.n_fft}, hop_length: {dataset.hop_length}, total_files: {len(dataset)}")
    General_logger.info(  f"[Train] Dataset => max_length: {dataset.max_length}, sr: {dataset.sr}, "f"n_fft: {dataset.n_fft}, hop_length: {dataset.hop_length}, total_files: {len(dataset)}")
    Evaluation_logger.info(f"[Eval] Dataset => max_length: {val_dataset.max_length}, sr: {val_dataset.sr}, "f"n_fft: {val_dataset.n_fft}, hop_length: {val_dataset.hop_length}, total_files: {len(val_dataset)}")
    General_logger.info(f"[Eval] Dataset => max_length: {val_dataset.max_length}, sr: {val_dataset.sr}, "f"n_fft: {val_dataset.n_fft}, hop_length: {val_dataset.hop_length}, total_files: {len(val_dataset)}")


    model = UNet(in_channels=1, out_channels=1).to(device)

    if load_model_path is not None:
        if os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path, map_location=device))
            print(f"Loaded model from {load_model_path}")
            train_logger.info(f"[Train] Loaded model from {load_model_path}")
        else:
            train_logger.warning(f"[Train] Model path {load_model_path} does not exist. Starting from scratch.")
            General_logger.warning(f"[Train] Model path {load_model_path} does not exist. Starting from scratch.")
        torch.cuda.empty_cache()
        gc.collect()
        train_logger.info("[Train] Cleared memory cache after loading model.")
        General_logger.info("[Train] Cleared memory cache after loading model.")


    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    train_logger.info(f"[Train] Starting => Batch_size={batch_size}, LR={learning_rate}, Epochs={epochs}")

    global loss_history_Epoches
    global loss_history_Batches

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            scaler = GradScaler()
            print(f"[train] Epoch {epoch + 1} / {epochs}")
            train_logger.info(f"[Train] Epoch {epoch+1}/{epochs} started.")

            for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
                if inputs is None or targets is None:
                    train_logger.warning(f"[Train] Skipping batch {batch_idx} due to None data.")
                    continue

                inputs, targets = inputs.to(device), targets.to(device)
                print(f"inputs: {inputs.shape}, targets: {targets.shape}")
        
                train_logger.debug(f"[Train] Epoch {epoch+1}, Batch {batch_idx} -> "f"inputs.shape={inputs.shape}, inputs.min={inputs.min():.4f}, inputs.max={inputs.max():.4f}; "f"targets.shape={targets.shape}, targets.min={targets.min():.4f}, targets.max={targets.max():.4f}")
                General_logger.debug(f"[Train] Epoch {epoch+1}, Batch {batch_idx} -> "f"inputs.shape={inputs.shape}, inputs.min={inputs.min():.4f}, inputs.max={inputs.max():.4f}; "f"targets.shape={targets.shape}, targets.min={targets.min():.4f}, targets.max={targets.max():.4f}")

                optimizer.zero_grad()

                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    train_logger.debug(f"[Train] outputs.shape={outputs.shape}, " f"outputs.min={outputs.min():.4f}, outputs.max={outputs.max():.4f}")
                    General_logger.debug(f"[Train] outputs.shape={outputs.shape}, " f"outputs.min={outputs.min():.4f}, outputs.max={outputs.max():.4f}")

                    combined_loss, l1_val, stft_val = criterion(outputs, targets)

       
                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()

         
                loss_history_Batches["l1"].append(l1_val.item())
                loss_history_Batches["spectral"].append(stft_val.item())
      
                loss_history_Batches["mse"].append(0.0)  # <--- valgfritt
                loss_history_Batches["combined"].append(combined_loss.item())


                loss_history_Epoches["l1"].append(l1_val.item())
                loss_history_Epoches["spectral"].append(stft_val.item())
                loss_history_Epoches["combined"].append(combined_loss.item())

     
                running_loss += combined_loss.item()
                train_logger.info(f"[Train] Epoch {epoch+1}, Batch {batch_idx}: "f"Loss={combined_loss.item():.6f}")

     
            epoch_loss = running_loss / len(dataloader)
            loss_history_Epoches["Runningloss"].append(running_loss)  
            avg_trainloss.append(epoch_loss)
            train_logger.info(f"[Train] Epoch {epoch+1} completed -> Avg Loss: {epoch_loss:.6f}")
            General_logger.info(f"[Train] Epoch {epoch+1} completed -> Avg Loss: {epoch_loss:.6f}")

            # Lagre checkpoint for hver epoke
            epoch_model_path = os.path.join(r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\CheckPoints",f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            train_logger.info(f"[Train] Saved checkpoint: {epoch_model_path}")
            General_logger.info(f"[Train] Saved checkpoint: {epoch_model_path}")
            General_logger.info(f"[Train] overall average loss: {avg_trainloss}")

    except Exception as e:
        train_logger.error(f"[Train] Error during training: {str(e)}")


    print("Creating Diagrams now")
    batches_figpath = os.path.join(diagramdirectory, "loss_curves_training_batches.png")
    epoch_figpath = os.path.join(diagramdirectory, "loss_curves_training_epoches.png")


    plot_loss_curves_Training_script_Batches(loss_history_Batches, out_path=batches_figpath)
    plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path=epoch_figpath)


    try:
        Evaluation_logger.info("[Eval] Starting Evaluation...")
        val_loss = evaluate_model(model.to(device), val_loader, device, output_dir)
        Evaluation_logger.info(f"[Eval] Validation Loss (final): {val_loss:.4f}")
        output_dir_eval = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Evaluation"
        print(f"val_loss after evaluation : {val_loss}")
        avg_sdr, avg_sir, avg_sar = evaluate_model_with_sdr_sir_sar(model=model, val_loader=val_loader, device=device,output_dir=output_dir_eval, sr=44100, n_fft=2048, hop_length=512)
        avgTotal = avg_sdr + avg_sir + avg_sar
        print(f"avgtotalloss after evaluation with sdr,sir,sar: {avgTotal}")
        totalloss = val_loss + avgTotal
        print(f"Avg sdr: {avg_sdr}, Avg sir: {avg_sir}, avg sar: {avg_sar}")
        print(f"Total loss before scheduler.step: {totalloss}")
        scheduler.step(totalloss)
    except Exception as e:
        Evaluation_logger.error(f"[Eval] Error during evaluation: {str(e)}")





    torch.cuda.empty_cache()
    gc.collect()
    train_logger.info("[Train] Training completed. Cleared memory cache.")
    General_logger.info("[Train] Training completed. Cleared memory cache.")



    Final_model_path = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\Final_model\Final_model2.pth"
    os.makedirs(os.path.dirname(Final_model_path), exist_ok=True)
    torch.save(model.state_dict(), Final_model_path)
    print(f"Final model saved at {Final_model_path}")
    train_logger.info(f"[Train] Final model saved at {Final_model_path}")
    General_logger.info(f"[Train] Final model saved at {Final_model_path}")

 
    print("Fine-tuning model now...")
    fine_tuned_model_path = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\Fine_tuned_model\Fine_tuned_model.pth"
    fine_tune_model(Final_model_path, fine_tuned_model_path, root_dir)

if __name__ == "__main__":
    train(load_model_path=None)
