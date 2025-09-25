import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from . import rope

from omegaconf import OmegaConf


def bias_dropout_add_scale(
    x: torch.Tensor, scale: torch.Tensor, residual: Optional[torch.Tensor], prob: float, training: bool
) -> torch.Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]

class TimeStepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(time: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                 [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding
    
    def forward(self, time: torch.Tensor):
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + (dropout_prob > 0), hidden_size)

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
    
    def token_drop(self, labels: torch.Tensor, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand((labels.shape[0], labels.shape[1]), device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels
    
    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids=None):
        B, L = labels.shape[0], labels.shape[1]
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings.reshape(B, -1)


class DDiTBlock(nn.Module):
    def __init__(
            self, 
            dim: int,
            n_heads: int, 
            cond_dim: int,
            mlp_ratio: int,
            dropout: float,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim 
        self.dropout = dropout

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, rotary_cos_sin: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = (
            item.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            for item in (q, k, v)
        )

        cos, sin = rotary_cos_sin
        original_dtype = q.dtype

        q = rope.apply_rotary_emb_torch(
            x=q.float(), cos=cos.float(), sin=sin.float()
        ).to(original_dtype)
        k = rope.apply_rotary_emb_torch(
            x=k.float(), cos=cos.float(), sin=sin.float()
        ).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v)) # (bsz, n_h, seq_len, h_dim)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask) # 原版没有dropout
        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
        x = bias_dropout_add_scale(
            self.attn_out(x), gate_msa, x_skip, prob=self.dropout, training=self.training
        )

        x = bias_dropout_add_scale(
            self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            gate_mlp, x, prob=self.dropout, training=self.training
        )

        return x

class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        
        self.norm_final1 = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)

        x = modulate(x=self.norm_final1(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x
    

class ConditionalDDiTlM(nn.Module):
    def __init__(self, n_tokens:int, num_classes:int, model_config: OmegaConf, pad_token_id=0):
        super().__init__()
        self.cfg = model_config
        self.n_tokens = n_tokens

        self.x_embed = nn.Embedding(n_tokens, model_config.d_model)
        self.t_embed = TimeStepEmbedder(model_config.d_cond)
        self.r_embed = rope.Rotary(dim=model_config.d_model // model_config.n_heads)
        self.c_embed = LabelEmbedder(num_classes, model_config.d_cond // 4, model_config.dropout)


        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=model_config.d_model,
                    n_heads=model_config.n_heads,
                    cond_dim=model_config.d_cond,
                    mlp_ratio=model_config.mlp_mult,
                    dropout=model_config.dropout,
                )
                for _ in range(model_config.n_layers)
            ] 
        )
        self.output_layer = DDiTFinalLayer(
            hidden_size=model_config.d_model,
            out_channels=n_tokens,
            cond_dim=model_config.d_cond
        )
        self.pad_token_id = pad_token_id
    
    def create_attn_mask(self, input_ids: torch.Tensor, padding_mask: torch.Tensor): 
        if padding_mask is None:
            attn_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        return attn_mask.bool()
    
    def forward(self, x, t, **kwargs):

        input_mask = kwargs.get("input_mask", None)
        c = kwargs.get("c", None)

        if input_mask is not None:
            x = x * input_mask
        
        attn_mask = self.create_attn_mask(x, padding_mask=input_mask)
        rotary_cos_sin = self.r_embed(x=x)

        x = self.x_embed(x)
        c = self.c_embed(c, self.training)
        t = self.t_embed(t)

        cond = t + c
        for blk in self.blocks:
            x = blk(x=x, rotary_cos_sin=rotary_cos_sin, c=cond, attn_mask=attn_mask)
        
        x = self.output_layer(x=x, c=cond)
        return x


