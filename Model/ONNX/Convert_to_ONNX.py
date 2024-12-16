import sys
import os
import torch
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)  # Use insert(0) to prioritize this path

from Model.Data.dataset import MUSDB18StemDataset
from Model.Model.model import UNet
import matplotlib.pyplot as plt




# Load the dataset
root_dir = r"C:\mappe1\musdb18"
dataset = MUSDB18StemDataset(root_dir=root_dir,subset="train")


# Convert tensor to NumPy for visualization
# Get one sample from the dataset
mixture_tensor, vocals_tensor = dataset[0]
mixture_np = mixture_tensor.squeeze().numpy()
plt.imshow(mixture_np, aspect="auto", origin="lower")
plt.title("Mixture Spectrogram")
plt.colorbar()
plt.show()
print(f"Shape of mixture tensor: {mixture_tensor.shape}")
print(f"Shape of vocals tensor: {vocals_tensor.shape}")




# Reshape the tensor for batch dimension
input_tensor = mixture_tensor.unsqueeze(0)




#Initialize the model
model = UNet(in_channels=1,out_channels=1)
model.load_state_dict(torch.load(r"C:\Users\didri\Desktop\AI AudioEnchancer\UNet_Model\unet_vocal_isolation.pth",weights_only=True))
model.eval()

#Save to ONNX
onnx_path = r"C:\Users\didri\Desktop\AI AudioEnchancer\Model\ONNX\ONNX_Model_files\Vocal_Isolation_UNet.onnx"
torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"Model converted to ONNX & saved as {onnx_path}")