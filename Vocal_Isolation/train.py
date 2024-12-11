import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from Vocal_Isolation.Data.dataset import MUSDB18StemDataset
from Vocal_Isolation.Model.model import UNet


def train():
    
    #Device that is used in training, PRIOR: GPU if not available using CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    
    
    #HyperParameters
    batch_size = 16
    learning_rate = 1e-4
    epochs = 50
    root_dir = r'C:\mappe1\musdb18'
    
    
    #Dataset and DataLoader
    dataset = MUSDB18StemDataset(root_dir=root_dir,subset='train')
    dataloader = DataLoader(dataset,batch_size=batch_size,suffle=True,num_workers=4)
    
    
    #Model, Loss, Optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    
    
    #Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs,targets) in enumerate(dataloader):
            inputs,targets = inputs.to(device), targets.to(device)
            
            #forward pass
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            
            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch +1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "unet_vocal_separation.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()