import torch
import gc
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Model.Logging.Logger import setup_logger
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')

def log_memory_usage(tag=""):

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_mem = torch.cuda.memory_allocated() / 1e9
        cached_mem = torch.cuda.memory_reserved() / 1e9
        train_logger.info(
            f"[Memory {tag}] GPU Memory: Total={total_mem:.2f} GB, "
            f"Allocated={allocated_mem:.2f} GB, Cached={cached_mem:.2f} GB"
        )
        print(
            f"[Memory {tag}] GPU Memory: Total={total_mem:.2f} GB, "
            f"Allocated={allocated_mem:.2f} GB, Cached={cached_mem:.2f} GB"
        )
    else:
        train_logger.info(f"[Memory {tag}] CUDA is not available. Using CPU only.")
        print(f"[Memory {tag}] CUDA is not available. Using CPU only.")


    cpu_mem = 0.0
    for obj in gc.get_objects():
        try:
            if hasattr(obj, "nbytes"):
                cpu_mem += obj.nbytes
        except Exception:

            continue

    cpu_mem /= 1e9  
    train_logger.info(f"[Memory {tag}] CPU Memory Usage: ~{cpu_mem:.2f} GB")
    print(f"[Memory {tag}] CPU Memory Usage: ~{cpu_mem:.2f} GB")
