import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb


# Need a function for tokenizer
# In here we also add CLS token and positional embedding
class Tokenizer(nn.Module): 
    def __init__(self, im_shape = torch.Size([3, 224, 224]),
                 patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        n_patches = int(np.prod(im_shape[1:]) // patch_size**2)
        self.embedding = torch.nn.Conv2d(im_shape[0],
                                         emb_size,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        
        self.cls = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_emb = nn.Parameter(torch.randn(1, n_patches + 1, emb_size))

    def forward(self, img):
        emb = self.embedding(img)
        emb = torch.permute(emb.view(emb.shape[0], emb.shape[1], -1), (0, 2, 1))
        emb = torch.cat([self.cls.expand(emb.shape[0], -1, -1), emb], dim=1)
        emb += self.positional_emb
        return emb


class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 expansion: int = 4,
                 drop_prob: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(expansion * emb_size, emb_size),
        )


class SkipConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 emb_size: int  =768,
                 num_heads: int = 8,
                 drop_prob: float =0.,
                 mask = None):
        super().__init__()
        self.emb_size = emb_size
        self.k_emb_size = emb_size // num_heads
        self.mask = mask
        self.num_heads = num_heads
        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        # split into mutliple heads
        bs = x.shape[0]
        seq_len = x.shape[1]
        queries = self.W_q(x).view(bs, -1, seq_len, self.k_emb_size)
        keys = self.W_k(x).view(bs, -1, seq_len, self.k_emb_size)
        values = self.W_v(x).view(bs, -1, seq_len, self.k_emb_size)

        # tranpose the get bs * seq_len * heads * emb_size
        queries.transpose(1, 2)
        keys.transpose(1, 2)
        values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.emb_size**0.5
        scores = self.dropout(F.softmax(scores, dim=-1))

        if self.mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, 0.0)

        hidden = torch.matmul(scores, values)
        concat = hidden.transpose(1, 2).contiguous().view(bs, -1, self.emb_size)
        return concat


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_prob: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_prob: float = 0.,
                 **kwargs):
        super().__init__(
            SkipConnection(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_prob)
            )),
            SkipConnection(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size,
                                 expansion=forward_expansion,
                                 drop_prob=forward_drop_prob),
                nn.Dropout(drop_prob)
            ))
        )

class Transformer(nn.Sequential):
    def __init__(self, depth: int = 4, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


tokenizer = Tokenizer()
img = torch.randn(2, 3, 224, 224)
toks = tokenizer(img)

transformer = Transformer()
out = transformer(toks)
print(out.shape)

