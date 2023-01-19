import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    
    seed = None
    if model_inputs.get('seed', None):
        seed = torch.Generator("cuda").manual_seed(model_inputs.get('seed', None))
        
    # Run the model
    with autocast("cuda"):
        image = model(generator=seed, **model_inputs)
        print(dir(image))
        image = image["sample"][0]
        
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}
