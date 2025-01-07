import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Model.Data.dataset import MUSDB18StemDataset,load_and_save_random_sample

dataset = MUSDB18StemDataset(root_dir=r"C:\mappe1\musdb18")
output_path = load_and_save_random_sample(output_dir=r"C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Data")

if output_path is not None:
    print(f"Random trimmed audio saved at: {output_path}")
else:
    print("Failed to save audio.")
