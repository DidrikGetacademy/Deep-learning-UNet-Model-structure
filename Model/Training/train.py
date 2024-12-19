import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)  
from Model.Data.dataset import MUSDB18StemDataset 
from Model.Model.model import UNet
from Model.Fine_tuning.Fine_Tuned_model import fine_tune_model

def train(load_model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    
    
    #HyperParameters
    batch_size = 3
    learning_rate = 1e-5
    epochs = 25
    root_dir = r'C:\mappe1\musdb18'
    
    
    #Dataset and DataLoader
    dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='train',
        sr=44100,
        n_fft=2048, 
        hop_length=512,
        max_length=1000,
        max_files=50,
        )
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True)
    
    
    
    val_dataset = MUSDB18StemDataset(
        root_dir=root_dir,
        subset='test',
        sr=44100,
        n_fft=2048, 
        hop_length=512, 
        max_length=1000,
        max_files=25 
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    
    

    model = UNet(in_channels=1, out_channels=1).to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path,weights_only=True))
        print(f"Loaded model from {load_model_path}")
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs,targets) in enumerate(dataloader):
            inputs,targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
            

                
  
            outputs = model(inputs.to(device))
            loss = criterion(outputs,targets.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
        
      
            
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
        
        
   
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
        

    fine_tuned_model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\PreTrained_model\fine_tuned_unet_vocal_isolation.pth"
    final_model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\PreTrained_model\unet_vocal_isolation_2.pth"
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Model saved.")
    print("starting fine tuning on model....")
    fine_tune_model(final_model_path,fine_tuned_model_path,root_dir)
    print("Fine tuning complete!")
 
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
    train(load_model_path=r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\PreTrained_model\unet_checkpoint_epoch30.pth") 