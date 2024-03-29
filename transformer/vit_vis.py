import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=1, embed_dim=768):
        super(PositionalEncoding, self).__init__()
        self.img_size = img_size
        H, W = img_size
        self.patch_size_h, self.patch_size_w = H // patch_size[0], W // patch_size[1]
        self.n_patches = self.patch_size_h * self.patch_size_w

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(
            x
        )  
        x = x.flatten(2)
        x = x.transpose(1, 2) 

        return x


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  
        dp = (
            q @ k_t
        ) * self.scale  
        attn = dp.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  
        weighted_avg = weighted_avg.flatten(2)  

        x = self.proj(weighted_avg) 
        x = self.proj_drop(x) 

        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        
        x = self.fc1(
            x
        )  
        x = self.act(x) 
        x = self.drop(x) 
        x = self.fc2(x)  
        x = self.drop(x)  

        return x

class EncoderLayer(nn.Module):

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttentionBlock(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = DecoderLayer(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        out, attn = self.attn(self.norm1(x))
        x = x + out
        x = x + self.mlp(self.norm2(x))

        return x, attn

class Transformer(nn.Module):

    def __init__(
            self,
            img_size=(64, 160),
            patch_size=(1, 10),
            in_channels=3,
            n_classes=2,
            embed_dim=6*4,
            depth=4,
            n_heads=4,
            mlp_ratio=4.,
            qkv_bias=False,
            p=0.3,
            attn_p=0.3, #init 0
    ):
        super().__init__()

        self.patch_embed = PositionalEncoding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                EncoderLayer(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        all_attention = []
        for block in self.blocks:
            x, attn = block(x)
            all_attention.append(attn)

        x = self.norm(x)

        cls_token_final = x[:, 0]  
        x = self.head(cls_token_final)

        return x, all_attention
