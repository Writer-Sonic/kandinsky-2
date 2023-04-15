import os
import torch
from kandinsky2 import get_kandinsky2


def download_model():
    # Use 'cuda' if available, otherwise use 'cpu'
    device = 0 if torch.cuda.is_available() else -1
    
    model = get_kandinsky2(device, task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)


    
if __name__ == "__main__":
    download_model()
