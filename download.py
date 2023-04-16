import os
import torch
from kandinsky2 import get_kandinsky2


def download_model():
    # Use 'cuda' if available, otherwise use 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu';
    
    model = get_kandinsky2(device, task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)


    
if __name__ == "__main__":
    download_model()
