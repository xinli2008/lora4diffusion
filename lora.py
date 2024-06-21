import torch
import torch.nn as nn
from config import *

class LoraLinearLayer(nn.Module):
    r"""
    A linear layer that is used with LORA
    Args:
        rank: the rank of the lora layer.
        alpha: the alpha of the lora layer. 
        input_features: number of input features.
        out_features: number of output features.
        dtype: the device to use for the layer's weights.
        device: the dtyle to use for the layer's weights.

    """
    def __init__(self, rank, alpha, in_features, out_features):
        super(LoraLinearLayer, self).__init__()

        self.down = nn.Linear(in_features, rank, bias = False)
        self.up = nn.Linear(rank, out_features, bias = False)

        self.rank = rank
        self.alpha = alpha
        self.in_faetures = in_features
        self.out_features = out_features
        
        nn.init.normal_(self.down.weight, std = 1 / self.rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Perform lora forward process
        Args:
            hidden_states:
        Return:
            torch.Tensor:   
        """
        original_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.alpha:
            up_hidden_states *= self.alpha / self.rank
        
        return up_hidden_states.to(original_dtype)
    
def inject_lora(model,name,layer):
    r"""
    Perform lora injection
    Args:
        i.e. model is unet, name is dec_convs.2.cross_attn.w_q
        and lora layer is Linear(in_features=64, out_features=32, bias=True)
    """
    name_cols=name.split('.')

    # 逐层下探到linear归属的module
    children=name_cols[:-1]
    cur_layer=model 
    # 这里找到倒数第二层, 比如说是cross_attention
    for child in children:
        cur_layer=getattr(cur_layer,child)
    
    lora_layer=LoraLinearLayer(rank, alpha, layer.in_features, layer.out_features)
    # 比如说将cross_attention的w_q设置为lora_linear layer
    setattr(cur_layer, name_cols[-1],lora_layer)