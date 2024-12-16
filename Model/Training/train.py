import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)  # Use insert(0) to prioritize this path
from Model.Data.dataset import MUSDB18StemDataset 
from Model.Model.model import UNet

def train(load_model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    
    
    #HyperParameters
    batch_size = 4
    learning_rate = 1e-5
    epochs = 5
    root_dir = r'C:\mappe1\musdb18'
    
    
    #Dataset and DataLoader
    dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='train',
        sr=44100,
        n_fft=1024, #Redce FFT size to save memory
        hop_length=256, #Smaller hop length will result in better time resolution
        max_length=10000,
        max_files=150 #Max amount of songs retrived from the dataset folder.
        )
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=6, pin_memory=True)
    
    
    
    val_dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='test',
        sr=44100,
        n_fft=1024, #Redce FFT size to save memory
        hop_length=256, #Smaller hop length will result in better time resolution
        max_length=10000,
        max_files=75 #Max amount of songs retrived from the dataset folder.
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    
    
    #Model, Loss, Optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path,weights_only=True))
        print(f"Loaded model from {load_model_path}")
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    
    #Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs,targets) in enumerate(dataloader):
            inputs,targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
            

                
            #forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs,targets.to(device))

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
            
            
            #monitor GPU memory
            if device == "cuda":
                gpu_memory_used = torch.cuda.memory_allocated() / 1024 ** 2 #Memory used in MB
                print(f"GPU memory Used: {gpu_memory_used:.2f} MB")
                
                if gpu_memory_used > 10000:
                    print("GPU memory usage is high. clearing cache...")
                    torch.cuda.empty_cache()
                    print(f"GPU memory Used: {gpu_memory_used:.2f} MB")
                    
            
            
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
        
        
        #Validation after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device,non_blocking=True),targets.to(device,non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs,targets)
                val_loss += loss.item()
                
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss / len(val_loader):.4f}")
        
        scheduler.step(val_loss / len(val_loader))
        
        #Checkpoint saving
        Checkpoint_dir = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\CheckPoints"
        checkpoint_path = os.path.join(Checkpoint_dir, f"unet_checkpoint_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(),checkpoint_path)    
        print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")
        


    #Save Final Model
    final_model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\unet_vocal_isolation.pth"
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Model saved.")
    
    #Testing on the entire test set
    test(model, val_loader, device)
    
    
def test(model,test_loader,device):
    print("Evaluating the model on the test set...")
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs,targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            test_loss += loss.item()
        print(f"Test Loss: {test_loss / len(test_loader):.4f}")
        print("Testing complete.")

if __name__ == "__main__":
    train(load_model_path=r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\unet_vocal_isolation.pth") #Passing saved model path if resuming training