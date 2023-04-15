import os
import torch
from kandinsky2 import get_kandinsky2


def download_model():
    model = get_kandinsky2('cuda', task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)

    
if __name__ == "__main__":
    download_model()
