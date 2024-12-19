import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
from torchvision.models import vgg16
import torchaudio


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


from Model.Data.dataset import MUSDB18StemDataset
from Model.Model.model import UNet


def freeze_encoder(model):
    for layer in model.encoder:
        for param in layer.parameters():
            param.requires_grad = False
    print("Encoder layers frozen for fine-tuning.")


class HybridLoss:
    def __init__(self, device):
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.device = device

        self.vgg = vgg16(pretrained=True).features[:5].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False




    def spectral_loss(self, output, target):
        return torch.mean((output - target) ** 2)




    def perceptual_loss(self, output, target):
        output_resized = F.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)
        target_resized = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        return self.mse_loss(self.vgg(output_resized), self.vgg(target_resized))



    def multi_scale_loss(self, output, target, scales=[1, 2, 4]):
        loss = 0
        for scale in scales:
            scaled_output = F.avg_pool2d(output, kernel_size=scale)
            scaled_target = F.avg_pool2d(target, kernel_size=scale)
            loss += self.mse_loss(scaled_output, scaled_target)
        return loss


    def stft_loss(self, pred_audio, target_audio, n_fft=2048, hop_length=512):
        pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        return self.mse_loss(torch.abs(pred_stft), torch.abs(target_stft))



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
        return total_loss
    
    
    def hybrid_loss(output, target):
        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()
        return 0.7 * criterion_mse(output,target) + 0.3 * criterion_l1(output,target)
    
    
    
def fine_tune_model(pretrained_model_path, fine_tuned_model_path, root_dir, batch_size=2, learning_rate=1e-5, fine_tune_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {device}")


    print("Loading dataset...")
    dataset = MUSDB18StemDataset(
        root_dir=root_dir, 
        subset='train', 
        sr=44100, 
        n_fft=2048, 
        hop_length=512,
        max_length=512, 
        max_files=50
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    print("Loading pretrained model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    freeze_encoder(model)  


    loss_functions = HybridLoss(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    
    print("Starting fine-tuning...")
    for epoch in range(fine_tune_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_functions.combined_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{fine_tune_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        scheduler.step(running_loss / len(dataloader))


    print("Saving fine-tuned model...")
    torch.save(model.state_dict(), fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")


if __name__ == "__main__":
    pretrained_model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\PreTrained_model\unet_vocal_isolation_2.pth"
    fine_tuned_model_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\PreTrained_model\fine_tuned_unet_vocal_isolation.pth"
    root_dir = r"C:\mappe1\musdb18" 

    fine_tune_model(pretrained_model_path, fine_tuned_model_path, root_dir)
