"""Estimators: a shared grid-space U-Net decoder, fed either by lifted image features or by a
rasterised pedestrian map. Output: [B, 200, 200] normalised cost in [0, 1] (label / max_cost_value)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BEV_N


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class GridUNet(nn.Module):
    """100 x 100 grid features -> 200 x 200 cost."""

    def __init__(self, cin: int, w: int = 32, out_ch: int = 1) -> None:
        super().__init__()
        self.out_ch = out_ch  # 1: cost in [0, 1]; 2: raw (mean logit, log-variance) for the probabilistic head
        self.e1 = block(cin, w)
        self.e2 = block(w, 2 * w, 2)       # 50
        self.e3 = block(2 * w, 4 * w, 2)   # 25
        self.d2 = block(4 * w + 2 * w, 2 * w)
        self.d1 = block(2 * w + w, w)
        self.up = nn.Sequential(nn.ConvTranspose2d(w, w // 2, 2, 2), nn.ReLU(inplace=True), nn.Conv2d(w // 2, out_ch, 3, 1, 1))

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        a = self.e1(g)
        b = self.e2(a)
        c = self.e3(b)
        x = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        x = self.d1(torch.cat([F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        if self.out_ch == 1:
            return torch.sigmoid(self.up(x)).squeeze(1)  # [B, 200, 200]
        return self.up(x)  # [B, out_ch, 200, 200]


class ImageEncoder(nn.Module):
    """RGB-D (or subset) image -> features at half resolution."""

    def __init__(self, cin: int, w: int = 32, out: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(block(cin, w, 2), block(w, 2 * w, 2), block(2 * w, 2 * w, 1))
        self.head = nn.Conv2d(2 * w + 2 * w, out, 1)
        self.skip = block(cin, 2 * w, 2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        deep = self.net(img)                                                  # 1/4
        deep = F.interpolate(deep, scale_factor=2, mode="bilinear", align_corners=False)  # 1/2
        return self.head(torch.cat([deep, self.skip(img)], 1))               # [B, out, H/2, W/2]


def scatter_mean(feat: torch.Tensor, idx: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """feat [B,C,h,w], idx [B,h,w] cell index or -1 -> mean features [B,C,n] and point count [B,1,n]."""
    B, C = feat.shape[:2]
    f = feat.flatten(2)                       # B,C,P
    i = idx.flatten(1)                        # B,P
    ok = i >= 0
    i = torch.where(ok, i, torch.zeros_like(i))
    w = ok.float().unsqueeze(1)               # B,1,P
    sums = torch.zeros(B, C, n, device=feat.device).scatter_add_(2, i.unsqueeze(1).expand(-1, C, -1), f * w)
    cnt = torch.zeros(B, 1, n, device=feat.device).scatter_add_(2, i.unsqueeze(1), w)
    return sums / cnt.clamp(min=1.0), cnt


class ImageToGrid(nn.Module):
    def __init__(self, cin: int, feat: int = 48, w: int = 32) -> None:
        super().__init__()
        self.enc = ImageEncoder(cin, out=feat)
        self.dec = GridUNet(feat + 1, w)

    def forward(self, x: dict) -> torch.Tensor:
        f = self.enc(x["image"])
        mean, cnt = scatter_mean(f, x["lift"], BEV_N * BEV_N)
        B = f.shape[0]
        g = torch.cat([mean, torch.log1p(cnt)], 1).view(B, -1, BEV_N, BEV_N)
        return self.dec(g)


class PositionToGrid(nn.Module):
    def __init__(self, w: int = 32, cin: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        # cin 3: position heatmap + heading; cin > 3 adds raster channels (position + interaction diagnostic)
        self.pre = nn.Sequential(nn.Conv2d(cin, 16, 3, 2, 1), nn.ReLU(inplace=True))  # 200 -> 100
        self.dec = GridUNet(16, w, out_ch)

    def forward(self, x: dict) -> torch.Tensor:
        return self.dec(self.pre(x["raster"]))


def build(inputs: str) -> nn.Module:
    if inputs in ("pos", "pos_fov"):
        return PositionToGrid()
    return ImageToGrid({"rgbd": 5, "depth": 2, "rgb": 3}[inputs])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
