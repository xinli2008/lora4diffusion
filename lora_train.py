import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import *
from dataset import train_dataset
from diffusion import forward_diffusion
from time_position_emb import TimePositionEmbedding
from unet import UNet
from lora import *

# 加载预训练好的底膜
state_dict = torch.load("/mnt/lora4diffusion/models/v1.1/model_epoch_200.pt")
model = UNet(1)
model.load_state_dict(state_dict)

# 遍历模块, 然后找到w_q, w_k, w_v, 并完成替换
for name, layer in model.named_modules():
    name_list = name.split(".")
    filter_layer_names = ["w_q", "w_k", "w_v"]
    if isinstance(layer, nn.Linear) and any(n in name_list for n in filter_layer_names):
        # i.e. model is uent and name is dec_convs.2.cross_attn.w_q
        inject_lora(model, name, layer)

model.to(device)

# 遍历模型的参数, 冻结非lora模块的参数
for name, layer in model.named_modules():
    if name.split(".")[-1] not in ["down", "up"]:
        layer.requires_grad = False
    else:
        layer.requires_grad = True

# 定义dataloader
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = True)

# 定义loss和优化器
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad== True, model.parameters()), lr=0.001)
loss_fn=nn.L1Loss()

def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch, device, writer):
    """
    训练模型一个周期。

    Args:
        model (nn.Module): 需要训练的模型。
        dataloader (DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        loss_fn (nn.Module): 损失函数。
        epoch (int): 当前训练的周期数。
        device (torch.device): 训练设备。
        writer (SummaryWriter): TensorBoard的SummaryWriter实例。

    Returns:
        float: 当前周期的平均损失。
    """
    model.train()
    epoch_loss = 0

    for step, (batch_x, batch_cls) in enumerate(dataloader):
        batch_x = batch_x.to(device) * 2 - 1

        # timestep
        batch_t = torch.randint(low=0, high=timestep, size=(batch_x.shape[0],)).to(device)

        # classification information
        batch_cls = batch_cls.to(device)

        # 前向扩散过程
        batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

        # 预测
        batch_predict_t = model(batch_x_t, batch_t, batch_cls)

        # 计算损失
        loss = loss_fn(batch_predict_t, batch_noise_t)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 将损失记录到 TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    return avg_loss

if __name__ == "__main__":
    # 创建目录以保存模型和日志文件
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 初始化 TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='logs')

    for epoch in range(max_epoch):
        print(f"Start training at epoch {epoch}")
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch, device, writer)
        print(f"End training at epoch {epoch}, Average Loss = {avg_loss:.4f}")

        # 保存模型
        if epoch % 10 == 0:
            model_save_path = os.path.join('models', f"model_epoch_{epoch}.pt")
            # 保存训练好的Lora权重
            lora_state={}
            for name,param in model.named_parameters():
                name_cols=name.split('.')
                filter_names=['down','up']
                if any(n==name_cols[-1] for n in filter_names):
                    lora_state[name]=param
            torch.save(lora_state, model_save_path)
            print(f"Model saved to {model_save_path}")

    writer.close()