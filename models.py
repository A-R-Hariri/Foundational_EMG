import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# ======== FSDP MODEL ========
# EMGPretrainer: variable-length, variable-channel input.
# Trained with VICReg + MAE objectives via pretrain_fsdp.py.
# Default dims: d_model=256, 4 heads, 4 transformer layers.

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, d_pe=64):
        super().__init__()
        self.d_pe = d_pe

    def forward(self, times_sec):
        B, L = times_sec.shape
        t = times_sec.unsqueeze(-1)
        i = torch.arange(self.d_pe // 2, device=times_sec.device, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (2 * i / self.d_pe))
        pe = t * freqs
        return torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)


class Encoder(nn.Module):
    """1D conv stem followed by a Transformer encoder.

    Accepts padded input (B, L, C) with accompanying time and channel validity masks.
    The strided conv reduces L to L', after which sinusoidal time encodings (derived
    from the original sample timestamps) are injected before the Transformer.
    """

    def __init__(self, max_C, d_model=256, kernel_size=8, stride=4, padding=4, d_pe=64):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.phi = nn.Conv1d(max_C, d_model, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True)
        self.pos_encoder = SinusoidalTimeEncoding(d_pe=d_pe)
        self.pe_proj = nn.Linear(d_model + d_pe, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)

    def forward(self, x, time_mask, ch_mask, times_sec, mae_mask=None):
        B, L, C = x.shape
        if ch_mask is not None:
            x = x * ch_mask.unsqueeze(1)
        x = x.transpose(1, 2)
        with autocast(device_type="cuda", enabled=False):
            x = self.phi(x.float())
        x = x.transpose(1, 2)
        L_new = x.size(1)
        with torch.no_grad():
            tm = F.max_pool1d(
                time_mask.float().unsqueeze(1),
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
            ).squeeze(1).bool()
            tm = self._fit_mask(tm, L_new)
        pe = self.pos_encoder(times_sec)
        pe = F.interpolate(pe.permute(0, 2, 1), size=L_new, mode='linear', align_corners=False).permute(0, 2, 1)
        x = self.pe_proj(torch.cat([x, pe], dim=-1))
        if mae_mask is not None:
            with torch.no_grad():
                mm = F.max_pool1d(
                    mae_mask.float().unsqueeze(1),
                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                ).squeeze(1).bool()
                mm = self._fit_mask(mm, L_new)
            tok = self.mask_token.view(1, 1, -1).expand(B, L_new, -1).to(x.dtype)
            x = torch.where(mm.unsqueeze(-1), tok, x)
        z_seq = self.transformer(x, src_key_padding_mask=~tm)
        return z_seq, tm

    @staticmethod
    def _fit_mask(mask, target_L):
        L = mask.size(1)
        if L == target_L:
            return mask
        if L > target_L:
            return mask[:, :target_L]
        pad = torch.zeros(mask.size(0), target_L - L, device=mask.device, dtype=torch.bool)
        return torch.cat([mask, pad], dim=1)


class VICHead(nn.Module):
    def __init__(self, d_model=256, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, z_seq, time_mask):
        denom = time_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        pooled = (z_seq * time_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.mlp(pooled)


class MAEDecoder(nn.Module):
    def __init__(self, max_C, d_model=256):
        super().__init__()
        self.proj = nn.Linear(d_model, max_C)

    def forward(self, z_seq, L_orig, C_max):
        y = self.proj(z_seq).transpose(1, 2)
        return F.interpolate(y, size=L_orig, mode='linear', align_corners=False).transpose(1, 2)


class EMGPretrainer(nn.Module):
    """VICReg + MAE self-supervised pretrainer for variable-length, multi-channel EMG.

    Architecture: strided conv1d stem -> Transformer encoder -> pooled VICReg head
                                                              -> interpolated MAE decoder

    Input: (B, L, C) padded float tensors with time_mask and ch_mask validity indicators.
    """

    def __init__(self, max_C, d_model=256):
        super().__init__()
        self.enc = Encoder(max_C=max_C, d_model=d_model)
        self.vic = VICHead(d_model=d_model, proj_dim=d_model)
        self.mae = MAEDecoder(max_C=max_C, d_model=d_model)

    def forward_encoder(self, x, time_mask, ch_mask, times_sec, mae_mask=None):
        return self.enc(x, time_mask, ch_mask, times_sec, mae_mask=mae_mask)

    def project_vic(self, z_seq, time_mask):
        return self.vic(z_seq, time_mask)

    def decode_mae(self, z_seq, L_orig, C_max):
        return self.mae(z_seq, L_orig, C_max)


# ======== DDP MODEL ========
# EMGTransformer: fixed-shape (B, L, C) input, conv stem + Transformer, VICReg only.
# Trained via pretrain_ddp.py on fixed-window-size data from data_pickles/.
# Default dims: d_model=128, 2 heads, 4 transformer layers.

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EMGTransformer(nn.Module):
    """Conv + Transformer encoder for fixed-shape EMG windows.

    Trims trailing zero-padding before processing, so batch windows with differing
    amounts of padding from different datasets are handled correctly.
    """

    def __init__(self, in_ch=8, d_model=128, nhead=2, num_layers=4,
                 dropout=0.1, max_len=100):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 128, kernel_size=4, stride=1, padding='same')
        self.conv2 = nn.Conv1d(128, d_model, kernel_size=2, stride=1, padding='same')
        self.norm = nn.LayerNorm(d_model)
        self.pos = PositionalEmbedding(max_len=max_len, d_model=d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self._trim_zeros(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.norm(x.transpose(1, 2))
        x = self.encoder(self.pos(x))
        return self.head(x.mean(dim=1))

    @staticmethod
    def _trim_zeros(x):
        valid = (x.abs().sum(dim=-1) != 0)
        max_len = int((x.size(1) - valid.flip(dims=[1]).float().argmax(dim=1)).max())
        return x[:, :max_len, :]
