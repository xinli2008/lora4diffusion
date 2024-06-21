import torch

image_size = 48
timestep = 1000
max_epoch = 500
batch_size = 256

# config for lora
rank = 8
alpha = 4

device = "cuda:3" if torch.cuda.is_available() else "cpu"