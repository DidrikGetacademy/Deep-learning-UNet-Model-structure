import torch
from Vocal_Isolation_Model.Model.model import UNet

def create_and_save_model(input_shape, in_channels=1, out_channels=1, save_path="unet_vocal_isolation.pth"):
    """
    Creates a U-Net model, validates it with a random tensor, and saves it.

    Args:
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).
        in_channels (int): Number of input channels for the U-Net model.
        out_channels (int): Number of output channels for the U-Net model.
        save_path (str): Path to save the model state dictionary.
    """
    # Select the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_shape = (1, 1, 256, 512)

    # Create a random input tensor on the selected device
    input_tensor = torch.randn(*input_shape, device=device)  # Example of a much smaller tensor

    print("Creating model...")

    # Initialize the U-Net model and move it to the selected device
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    print("Model initialized.")

    # Print memory usage (if using GPU)
    if device.type == "cuda":
        print("Allocated memory:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Max allocated memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

    # Perform a forward pass
    try:
        output = model(input_tensor)
        print("Forward pass successful.")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print("Error during forward pass:", str(e))
        return

    # Save the model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at '{save_path}'.")

if __name__ == "__main__":
    # Define input tensor shape
    input_shape = (1, 1, 256, 512)  # Adjusted for smaller input size during testing

    # Create, validate, and save the model
    create_and_save_model(input_shape, in_channels=1, out_channels=1, save_path="unet_vocal_isolation.pth")
