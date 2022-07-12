FROM stylegan3:latest

ENV TZ=Europe/Berlin

RUN pip install denoising-diffusion-pytorch