"""2D U-Net and Attention U-Net for FLAIR abnormality segmentation.

Both use Monte-Carlo dropout so that predictive uncertainty can be estimated
at inference time (set ``model.mc_dropout = True`` and run several passes)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Dropout2d):
    def __init__(self, p, owner):
        super().__init__(p)
        self._owner = [owner]   # list avoids registering the parent as a sub-module

    def forward(self, x):
        return F.dropout2d(x, self.p, training=self.training or self._owner[0].mc_dropout)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, p, owner):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.drop = MCDropout(p, owner) if p > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.net(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018): the decoder feature g acts as
    the query that weights the encoder skip feature x."""

    def __init__(self, c_g, c_x, c_int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(c_g, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.wx = nn.Sequential(nn.Conv2d(c_x, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.last_alpha = None

    def forward(self, g, x):
        alpha = self.psi(F.relu(self.wg(g) + self.wx(x), inplace=True))
        self.last_alpha = alpha.detach()
        return x * alpha


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=16, attention=False, p=0.2):
        super().__init__()
        self.mc_dropout = False
        self.attention = attention
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([ConvBlock(in_ch, c[0], 0, self), ConvBlock(c[0], c[1], 0, self),
                                  ConvBlock(c[1], c[2], 0, self), ConvBlock(c[2], c[3], p, self)])
        self.bottleneck = ConvBlock(c[3], c[4], p, self)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(3, -1, -1)])
        self.gates = nn.ModuleList([AttentionGate(c[i], c[i], max(c[i] // 2, 8)) for i in range(3, -1, -1)]) \
            if attention else None
        self.dec = nn.ModuleList([ConvBlock(c[i] * 2, c[i], p if i >= 2 else 0, self) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x, return_bottleneck=False):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        b = self.bottleneck(x)
        x = b
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[3 - j]
            if self.gates is not None:
                s = self.gates[j](x, s)
            x = dec(torch.cat([x, s], dim=1))
        logits = self.head(x)
        return (logits, b) if return_bottleneck else logits


def build(name):
    if name == "unet":
        return UNet(attention=False)
    if name == "attention_unet":
        return UNet(attention=True)
    raise ValueError(name)
