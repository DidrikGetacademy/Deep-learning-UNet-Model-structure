import torch 
from Vocal_Isolation_Model.Model.model import UNet # Imports the model
from Vocal_Isolation_Model.Data.dataset import MUSDB18Dataset
import matplotlib.pyplot as plt


# Load the dataset
root_dir = r"C:\Users\didri\Downloads\musdb18"
dataset = MUSDB18Dataset(root_dir=root_dir,subset="train")


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
model.load_state_dict(torch.load("vocal_isolation_unet.pth"))
model.eval()

#Save to ONNX
onnx_path = "Vocal_Isolation_UNet.onnx"
torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_name=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"Model converted to ONNX & saved as {onnx_path}")