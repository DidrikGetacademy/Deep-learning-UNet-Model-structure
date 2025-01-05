import torch
import tensorflow as tf
print("CUDA Available:", torch.cuda.is_available())
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
