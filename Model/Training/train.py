import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import os
import sys
import gc
from torch import autocast, GradScaler
from datasets import load_dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
import multiprocessing
from Model.Logging.Logger import setup_logger
from Model.Data.dataset import MUSDB18StemDataset
from Model.Data.dataset import VCTKSpectrogram
from Model.Model.model import UNet
from Model.Training.Evaluation import evaluate_model, evaluate_model_with_sdr_sir_sar
from Model.Training.Fine_Tuned_model import fine_tune_model

# Viktig: Oppdatert import av plot-funksjoner
from Model.Training.Loss_Diagram_Values import (
    plot_loss_curves_Training_script_epoches,
    plot_loss_curves_Training_script_Batches,
    plot_loss_curves_evaluation,

)

train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached GPU Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA is not available.")

num_workers = multiprocessing.cpu_count() // 2

loss_history_Epoches = {
    "l1": [],
    "spectral": [],
    "combined": [],
    "Total_loss_per_epoch": [],
    "avg_trainloss":[],
}

loss_history_Batches = {
    "l1": [],
    "spectral": [],
    "mse": [],
    "combined": [],
}

avg_trainloss = {"epoch_loss":[],}



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
        hop_length = 1024
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
    
            
        stft_loss /= batch_size
        scaled_stft_loss = stft_loss / 1000.0
        combined_loss = 0.3 * l1 + 0.7 * scaled_stft_loss

        train_logger.debug( f"[HybridLoss] L1={l1.item():.6f}, STFT={stft_loss.item():.6f} "f"(scaled={scaled_stft_loss.item():.6f}), Combined={combined_loss.item():.6f}")
        return combined_loss, l1, stft_loss








def train(load_model_path=r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\Fine_tuned_model\Fine_tuned_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"[Train] Using device: {device}")
    print(f"Device: {device}")


    batch_size = 8
    learning_rate = 1e-5
    epochs = 5
    root_dir = r'C:\mappe1\musdb18'
    


    diagramdirectory = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Diagram_Resultater"
    os.makedirs(diagramdirectory, exist_ok=True)

 
    output_dir = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Evaluation"
    os.makedirs(output_dir, exist_ok=True)

    
    musdb18_dataset  = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='train',
        sr=44100,
        n_fft=2048,
        hop_length=1024,
        max_length=176400,
        max_files=100,
    )
   

    vctk_dataset = load_dataset("speechcolab/vctk")["train"]
    vctk_spectrogram_dataset = VCTKSpectrogram(dataset=vctk_dataset, n_fft=2048, hop_length=1024, max_length=176400)

    combined_dataset = ConcatDataset([musdb18_dataset,vctk_spectrogram_dataset])

    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='test',
        sr=44100,
        n_fft=2048,
        hop_length=1024,
        max_length=176400,
        max_files=40,
        
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_logger.info(f"[Train] MUSDB18 Dataset => total_files: {len(musdb18_dataset)}")
    train_logger.info(f"[Train] VCTK Dataset => total_files: {len(vctk_spectrogram_dataset)}")
    train_logger.info(f"[Val] Validation Dataset => total_files: {len(val_dataset)}")



    model = UNet(in_channels=1, out_channels=1).to(device)

    if load_model_path is not None:
        if os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path, map_location=device))
            print(f"Loaded model from {load_model_path}")
            train_logger.info(f"[Train] Loaded model from {load_model_path}")
        else:
            train_logger.warning(f"[Train] Model path {load_model_path} does not exist. Starting from scratch.")

        torch.cuda.empty_cache()
        gc.collect()
        train_logger.info("[Train] Cleared memory cache after loading model.")

    global loss_history_Epoches
    global loss_history_Batches
    global avg_trainloss

    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.6)
    train_logger.info(f"[Train] Starting => Batch_size={batch_size}, LR={learning_rate}, Epochs={epochs}")

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

                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device,non_blocking=True)
                print(f"inputs: {inputs.shape}, targets: {targets.shape}")
        
                train_logger.debug(f"[Train] Epoch {epoch+1}, Batch {batch_idx} -> "f"inputs.shape={inputs.shape}, inputs.min={inputs.min():.4f}, inputs.max={inputs.max():.4f}; "f"targets.shape={targets.shape}, targets.min={targets.min():.4f}, targets.max={targets.max():.4f}")

                accumulation_steps = 4  
                optimizer.zero_grad()
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    train_logger.debug(f"[Train] outputs.shape={outputs.shape}, " f"outputs.min={outputs.min():.4f}, outputs.max={outputs.max():.4f}")

                    combined_loss, l1_val, stft_val = criterion(outputs, targets)

       
                scaler.scale(combined_loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                   scaler.step(optimizer)
                   optimizer.zero_grad()
                   scaler.update()
                   torch.cuda.empty_cache()
                   gc.collect() 
         
                loss_history_Batches["l1"].append(l1_val.item())
                loss_history_Batches["spectral"].append(stft_val.item())
                loss_history_Batches["combined"].append(combined_loss.item())


                loss_history_Epoches["l1"].append(l1_val.item())
                loss_history_Epoches["spectral"].append(stft_val.item())
                loss_history_Epoches["combined"].append(combined_loss.item())

     
                running_loss += combined_loss.item()
                train_logger.info(f"[Train] Epoch {epoch+1}, Batch {batch_idx}: "f"Loss={combined_loss.item():.6f}")

     
            epoch_loss = running_loss / len(dataloader)
            loss_history_Epoches["Total_loss_per_epoch"].append(running_loss)  
            avg_trainloss["epoch_loss"].append(epoch_loss)


            # Lagre checkpoint for hver epoke
            epoch_model_path = os.path.join(r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\CheckPoints",f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            train_logger.info(f"[Train] Epoch {epoch+1} completed -> Avg Loss: {epoch_loss:.6f}")
            train_logger.info(f"[Train] Saved checkpoint: {epoch_model_path}")
    

    except Exception as e:
        train_logger.error(f"[Train] Error during training: {str(e)}")


    print("Creating Diagrams now")
    batches_figpath = os.path.join(diagramdirectory, "loss_curves_training_batches.png")
    epoch_figpath = os.path.join(diagramdirectory, "loss_curves_training_epoches.png")
    print("creating Training plot Curves..")
    plot_loss_curves_Training_script_Batches(loss_history_Batches, out_path=batches_figpath)
    plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path=epoch_figpath)


    try:
        train_logger.info("[Eval] Starting evaluate_model")
        print("Starting Evaluation now.")
        val_loss = evaluate_model( model.to(device),  val_loader,device, output_dir )
        train_logger.info(f"[Eval] Validation Loss (final): {val_loss:.4f}")
        print(f"[Eval] Validation Loss (final): {val_loss:.4f}")
        
 


        output_dir_eval = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Evaluation"
        train_logger.info("[EVAL] starting  evaluate_model_with_sdr_sir_sar.")
        avg_sdr, avg_sir, avg_sar = evaluate_model_with_sdr_sir_sar(model=model, dataloader=val_loader, output_csv_path=os.path.join(output_dir_eval, "Evaluation_results.csv"),loss_diagram_func=plot_loss_curves_evaluation ,  device=device,  output_dir=output_dir_eval,  sr=44100, n_fft=2048, hop_length=1024)



        avgTotal = avg_sdr + avg_sir + avg_sar
        totalloss = val_loss + avgTotal
        train_logger.info(f"Avg SDR: {avg_sdr}, Avg SIR: {avg_sir}, avg SAR: {avg_sar}")
        train_logger.info(f"Total loss (VAL_LOSS + SDR/SIR/SAR): {totalloss}")
        print(f"Avg SDR: {avg_sdr}, Avg SIR: {avg_sir}, avg SAR: {avg_sar}")
        print(f"Total loss (VAL_LOSS + SDR/SIR/SAR): {totalloss}")
        



         #sheduler oppdaterer læringsrate basert på total loss.
        scheduler.step(totalloss)

    except Exception as e:
        train_logger.error(f"[Eval] Error during evaluation: {str(e)}")
        train_logger.info(f"Error during evaluation.")
        print(f"[Eval] Error during evaluation: {str(e)}")
        print(f"Error during evaluation.")


    torch.cuda.empty_cache()
    gc.collect()
    train_logger.info("[Train] Training completed. Cleared memory cache.")
    print("[Train] Training completed. Cleared memory cache.")



    Final_model_path = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\Final_model\Final_model.pth"
    os.makedirs(os.path.dirname(Final_model_path), exist_ok=True)
    torch.save(model.state_dict(), Final_model_path)
    print(f"Final model saved at {Final_model_path}")
    train_logger.info(f"[Train] Final model saved at {Final_model_path}")
    print(f"[Train] Final model saved at {Final_model_path}")


 
    print("Fine-tuning model now...")
    train_logger.info("Fine-tuning model now...")
    fine_tuned_model_path = r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\UNet_Model\Fine_tuned_model\Fine_tuned_model.pth"
    fine_tune_model(Final_model_path, fine_tuned_model_path, root_dir)

if __name__ == "__main__":
    train(load_model_path=None)
