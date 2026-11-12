import torch
from tools.registery import LOSS_REGISTRY
from torch.nn import functional as F

import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Attention'
        self.masks = []
        self.sparsity_ratios = []
        self.params = {}
        self.layer_counter = 0

    def info(self):
        return {
            "name": self.name,
            "params": self.params,
            "masks": self.masks,
            "sparsity": self.calculate_average_sparsity_ratio(self.sparsity_ratios)
        }

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def maybe_save_mask(self, attention_mask, modulo_layers=8):
        if self.layer_counter % modulo_layers == 0:
            self.masks.append(attention_mask[0, 0].clone().cpu().numpy())

        self.layer_counter += 1

    def forward(self, queries, keys, values, *args, **kwargs):
        """Base forward method for prefilling attention patterns.
        
        Args:
            queries: (batch_size, num_heads, seq_len, head_dim)
            keys: (batch_size, num_heads, seq_len, head_dim)
            values: (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            attention_output: (batch_size, num_heads, seq_len, head_dim)
        """
        raise NotImplementedError

    def generation_forward(self, prefilling_queries, prefilling_keys, prefilling_values,
                         generation_queries, generation_keys, generation_values, *args, **kwargs):
        """Base forward method for generation attention patterns.
        
        Args:
            prefilling_queries: Queries from prefilling stage
            prefilling_keys: Keys from prefilling stage
            prefilling_values: Values from prefilling stage
            generation_queries: New queries for generation
            generation_keys: New keys for generation
            generation_values: New values for generation
            
        Returns:
            attention_output: Output for generation tokens
        """
        raise NotImplementedError

    @staticmethod
    def attention(queries, keys, values, attention_mask, return_attention_scores=False):
        """Standard attention computation."""
        attention_weights = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(queries.size(-1))
        attention_weights += attention_mask.to(queries.dtype) * torch.finfo(queries.dtype).min
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attention_output = torch.matmul(attention_weights, values)
        if return_attention_scores:
            return attention_output, attention_weights
        return attention_output
    
    @staticmethod
    def get_causal_mask(seq_len, device):
        """Creates a causal mask where future tokens cannot attend to past tokens.
        
        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, seq_len, seq_len) where True/1 indicates
                 that position (i,j) should be masked (set to -inf before softmax)
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def get_local_mask(seq_len, window_size, device):
        """Creates a local attention mask where tokens can only attend to nearby tokens
        within a fixed window size, plus the causal constraint.
        
        Args:
            seq_len: Length of the sequence
            window_size: Size of the local attention window including current token
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, seq_len, seq_len) where True/1 indicates
                 that position (i,j) should be masked (set to -inf before softmax)
        """
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=-(window_size-1))
        mask = torch.tril(mask, diagonal=0)
        mask = (~mask)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def get_generation_mask(gen_len, prefill_len, device):
        """Creates a mask that allows generation tokens to:
        1. Attend to all prefilling tokens
        2. Attend causally to other generation tokens
        
        Args:
            gen_len: Number of tokens being generated
            prefill_len: Number of tokens in the prefill context
            device: Device to create the mask on
            
        Returns:
            mask: Boolean tensor of shape (1, 1, gen_len, prefill_len + gen_len)
                 where True indicates positions that should be masked
        """
        return torch.cat([
            torch.zeros((1, 1, gen_len, prefill_len), dtype=torch.bool, device=device),
            Attention.get_causal_mask(gen_len, device)
        ], dim=-1)

    @staticmethod
    def calculate_sparsity_ratio(mask):
        """Calculates the sparsity ratio of an attention mask.

        This method computes what fraction of the possible attention connections are masked 
        assuming that attention is causal, i.e., that tokens cannot attend to tokens before them.
        A higher ratio means more sparse attention. Asummes batch_size = 1.

        Args:
            mask: Boolean tensor of shape (batch_size, num_heads, queries_len, keys_len) where
                 True/1 indicates masked (disabled) attention connections

        Returns:
            float: The sparsity ratio between 0 and 1, where:
                  0 means all possible connections are enabled (dense attention)
                  1 means all possible connections are masked (completely sparse)
        """

        _, _, queries_len, keys_len = mask.shape

        if queries_len != keys_len:
            prefill_length = keys_len - queries_len
            total_connections = queries_len * (queries_len + 1) // 2 + prefill_length * queries_len
        else:
            total_connections = queries_len * (queries_len + 1) // 2

        connections_per_head = (~mask).long().sum(dim=(-1, -2))
        non_masked_ratio = (connections_per_head.float() / total_connections).mean(dim=-1).item()
        sparsity_ratio = 1 - non_masked_ratio

        return sparsity_ratio
    
    @staticmethod
    def calculate_average_sparsity_ratio(sparsity_ratios):
        return sum(sparsity_ratios) / len(sparsity_ratios) if sparsity_ratios else 0
    
@LOSS_REGISTRY.register()
class l1_loss():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = torch.nn.L1Loss()

    def forward(self, x, y):
        if not self.as_loss:
            with torch.no_grad():
                return self.weight*self.loss_item(x, y)

        else:
            return self.weight*self.loss_item(x, y)


@LOSS_REGISTRY.register()
class psnr():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = torch.nn.MSELoss()

    def forward(self, x, y, data_range=1.):
        if not self.as_loss:
            with torch.no_grad():
                mse = self.loss_item(x.clamp(0, 1.), y.clamp(0., 1.))
                return 10*torch.log10(data_range**2/(mse+1e-14))
        else:
            mse = self.loss_item(x, y)
            return 10*torch.log10(data_range**2/mse)

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target)**2 + eps))

@LOSS_REGISTRY.register()
class Charbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss

    def forward(self, x, y):
        if not self.as_loss:
            with torch.no_grad():
                loss = self.loss_item(x, y)
        else:
            loss = self.loss_item(x, y)
        return loss
    
@LOSS_REGISTRY.register()
class HalfCharbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss

    def forward(self, x, y):
        if len(x.shape) -- 5:
            N, T, C, H, W = x.shape
            x, y = F.interpolate(x.view(-1, C, H, W), scale_factor=0.5, mode='bilinear'), F.interpolate(y.view(-1, C, H, W), scale_factor=0.5, mode='bilinear')
        if not self.as_loss:
            with torch.no_grad():
                loss = self.loss_item(x, y)
        else:
            loss = self.loss_item(x, y)
        return loss
    
@LOSS_REGISTRY.register()
class GroupCharbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss

    def forward(self, xin, y):
        loss = 0.
        for idx in range(len(self.weight)):
            x = xin[idx]
            h, w = x.shape[2:]
            ycur = F.interpolate(y, (h, w), mode='bilinear')
            if not self.as_loss:
                with torch.no_grad():
                    loss += self.loss_item(x, ycur)*self.weight[idx]
            else:
                loss += self.loss_item(x, ycur)*self.weight[idx]
        return loss

@LOSS_REGISTRY.register()
class ECharbonier():
    def __init__(self, loss_dict):
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss
        self.emask_type = loss_dict.emask_type

    def forward(self, x, y, e):
        e_mask = torch.sum(torch.abs(e), 1, keepdim=True)
        if self.emask_type == 'flag':
            e_mask = e_mask/torch.max(e_mask, torch.tensor(1e-5).to(e.device))
        loss = self.loss_item(x, y)
        loss_e = self.loss_item(x*e_mask, y*e_mask)*self.weight
        return loss+loss_e
    
@LOSS_REGISTRY.register()
class SmoothLoss():
    def __init__(self, loss_dict):
        self.weight = loss_dict.weight
        
    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy
        
    def cal_grad2_error(self, flow, img):
        flow = flow / 20.
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
        return error / 2.0

    def forward(self, flow, gt):
        loss = self.weight*self.cal_grad2_error(flow, gt)
        return loss.mean()
    
    
@LOSS_REGISTRY.register()
class AnnealCharbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.fact = loss_dict.fact
        self.loss_item = charbonnier_loss

    def forward(self, x, y, x0, step):
        if not self.as_loss:
            with torch.no_grad():
                loss0 = self.loss_item(x, y)
                loss1 = self.loss_item(x0, y)
        else:
            loss0 = self.loss_item(x, y)
            loss1 = self.loss_item(x0, y)
        curfact = self.fact**step
        loss = loss0*(1-curfact)+loss1*curfact
        return loss*self.weight