import torch
from torch import autocast
from kandinsky2 import get_kandinsky2
import base64
from io import BytesIO
import os

def init():
    global model
    device = 0 if torch.cuda.is_available() else -1
    model = get_kandinsky2(device, task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)
                           
def inference(model_inputs:dict):
    global model

    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('steps', 100)
    batch_size = model_inputs.get('batch_size', 1)
    guidance_scale = model_inputs.get('guidance_scale', 4)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    with autocast("cuda"):
        images = model.generate_text2img(
            prompt,
            num_steps=steps,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            h=height,
            w=width,
            sampler='p_sampler',
            prior_cf_scale=4,
            prior_steps="5"
        )
        image = images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}
