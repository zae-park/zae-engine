import torch
import torch.nn as nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class TransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        depth,
        heads,
        mlp_dim,
        tmp_ch,
        pool="cls",
        channels=3,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
        resnetlayers=8,
        resnet_kernelsize=25,
    ):
        super().__init__()
        patch_height = 1  # patch size
        self.patch_size = patch_height
        resnet_filnal_channels = tmp_ch  # final channel
        dim = resnet_filnal_channels * patch_height

        img_size = image_size  # temporary ( final resolution)
        num_patches = img_size // patch_height

        self.to_patch_embedding = Rearrange("b c (w p1) -> b (w) (p1 c)", p1=patch_height)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim, depth=depth, heads=heads, dim_head=8, mlp_dim=mlp_dim, dropout=0.1)
        pool = "cls"
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        b, n, l = x.shape

        while True:
            if l % self.patch_size != 0:
                out = x[:, :, 0:-1]
                b, n, l = x.shape
            else:
                break

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MLP(nn.Module):
    def __init__(self, ch_list):
        super(MLP, self).__init__()
        self.ch_list = ch_list
        self.relu = nn.ReLU()
        self.mlp = self._stack_layers()

    def _stack_layers(self):
        layers = []
        num_layers = len(self.ch_list)
        if num_layers < 2:
            return nn.Sequential(*layers)
        else:
            for i_ch in range(num_layers - 1):
                layers.append(
                    nn.Conv1d(
                        self.ch_list[i_ch],
                        self.ch_list[i_ch + 1],
                        kernel_size=1,
                        bias=False,
                    )
                )
                if i_ch != num_layers - 1:
                    layers.append(self.relu)
            return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention2(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention2(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
