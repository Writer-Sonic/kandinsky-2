FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install required system packages
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python packages
RUN pip3 install \
    torch==1.13.1 \
    sentencepiece==0.1.97 \
    accelerate==0.16.0 \
    Pillow==9.5.0 \
    attrs==22.2.0 \
    opencv-python==4.7.0.72 \
    tqdm==4.65.0 \
    ftfy==6.1.1 \
    blobfile==2.0.1 \
    transformers==4.23.1 \
    torchvision==0.14.1 \
    omegaconf==2.3.0 \
    pytorch_lightning==2.0.1 \
    einops==0.6.0 \
    git+https://github.com/openai/CLIP.git

# Add requirements.txt and install additional dependencies if needed
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD server.py .
EXPOSE 8000



ADD download.py .
RUN python3 download.py

ADD app.py .
CMD python3 -u server.py
