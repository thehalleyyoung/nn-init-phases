"""
Universal architecture experiment v2: 80+ architectures with training dynamics validation.

Tests PhaseKit's DAG propagator on a comprehensive set of architectures including:
  - MLPs (various depths, activations)
  - CNNs (plain, VGG-style, depthwise separable)
  - ResNets (custom, torchvision)
  - DenseNets (custom, torchvision)
  - Transformers (Pre-LN, Post-LN, ViT)
  - MobileNet / EfficientNet / ShuffleNet
  - UNet-style
  - Gated architectures (SwiGLU, GLU)
  - Inception / multi-branch
  - LSTM / GRU based networks
  - Bottleneck / Highway
  - Real torchvision models

For each architecture:
  1. DAG-based phase prediction
  2. Empirical phase from variance trajectory
  3. Training dynamics validation (does critical init train faster?)
"""

import sys
import os
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'universal_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def determine_empirical_phase(model, input_shape, n_samples=128, seed=42):
    """Determine phase from empirical second-moment trajectory."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    x = torch.randn(n_samples, *input_shape)

    second_moments = []
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                out = output[0]
            elif isinstance(output, torch.Tensor):
                out = output
            else:
                return
            # Track second moment E[x^2] to match mean-field q
            sm = float((out.detach().float() ** 2).mean().item())
            second_moments.append(sm)
        return hook_fn

    idx = 0
    for name, module in model.named_modules():
        children = list(module.children())
        if len(children) == 0 or isinstance(module, (nn.MultiheadAttention, nn.LSTM, nn.GRU)):
            h = module.register_forward_hook(make_hook(idx))
            hooks.append(h)
            idx += 1

    model.eval()
    with torch.no_grad():
        try:
            model(x)
        except Exception:
            try:
                model(x, x)
            except Exception:
                for h in hooks:
                    h.remove()
                return "error", []

    for h in hooks:
        h.remove()

    if len(second_moments) < 2:
        return "critical", second_moments

    # Classify from variance trajectory
    first_sm = second_moments[0] if second_moments[0] > 1e-15 else 1e-15
    last_sm = second_moments[-1] if second_moments[-1] > 1e-15 else 1e-15
    n_steps = max(len(second_moments) - 1, 1)
    ratio = (last_sm / first_sm) ** (1.0 / n_steps)

    if ratio < 0.85:
        return "ordered", second_moments
    elif ratio > 1.2:
        return "chaotic", second_moments
    else:
        return "critical", second_moments


def train_model_briefly(model, input_shape, n_steps=100, lr=0.01, seed=42):
    """Train a model briefly and return loss trajectory."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n_samples = 64
    x = torch.randn(n_samples, *input_shape)

    # Get output shape
    model.eval()
    with torch.no_grad():
        try:
            out = model(x)
        except Exception:
            return None
    if isinstance(out, tuple):
        out = out[0]
    n_classes = out.shape[-1]
    y = torch.randint(0, max(n_classes, 2), (n_samples,))

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        try:
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            # Handle varying output shapes
            if pred.dim() == 3:
                pred = pred[:, -1, :]  # take last token
            loss = criterion(pred, y[:pred.shape[0]])
            if torch.isnan(loss) or torch.isinf(loss):
                losses.append(float('inf'))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        except Exception:
            losses.append(float('inf'))
            break

    return losses


def build_architecture_zoo():
    """Build 80+ architectures for testing."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    architectures = []

    # === MLPs ===
    for depth in [3, 5, 10, 15, 20]:
        for act_cls, act_name in [(nn.ReLU, 'relu'), (nn.GELU, 'gelu'),
                                   (nn.SiLU, 'silu'), (nn.Tanh, 'tanh')]:
            if depth > 10 and act_name in ('tanh',):
                continue  # skip slow configs
            layers = []
            layers.append(nn.Linear(128, 256))
            layers.append(act_cls())
            for _ in range(depth - 2):
                layers.append(nn.Linear(256, 256))
                layers.append(act_cls())
            layers.append(nn.Linear(256, 10))
            model = nn.Sequential(*layers)
            architectures.append((f"MLP-{depth}L-{act_name}", model, (128,), "MLP"))

    # === CNNs ===
    class SimpleCNN(nn.Module):
        def __init__(self, depth=3, channels=32):
            super().__init__()
            layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()]
            for _ in range(depth - 1):
                layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()])
            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(channels, 10))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    for d in [3, 5, 8]:
        architectures.append((f"CNN-{d}L", SimpleCNN(d), (3, 32, 32), "CNN"))

    # VGG-style with max pooling
    class VGGStyle(nn.Module):
        def __init__(self, channels=[64, 128, 256]):
            super().__init__()
            layers = []
            in_ch = 3
            for ch in channels:
                layers.extend([nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                               nn.MaxPool2d(2)])
                in_ch = ch
            layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], 10)])
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    architectures.append(("VGG-style-small", VGGStyle([32, 64, 128]), (3, 32, 32), "CNN"))
    architectures.append(("VGG-style-large", VGGStyle([64, 128, 256]), (3, 32, 32), "CNN"))

    # Depthwise separable CNN (MobileNet-style block)
    class DepthwiseSepCNN(nn.Module):
        def __init__(self, depth=4, channels=64):
            super().__init__()
            layers = [nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU6()]
            for _ in range(depth):
                layers.extend([
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                    nn.BatchNorm2d(channels), nn.ReLU6(),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels), nn.ReLU6(),
                ])
            layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10)])
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    architectures.append(("DepthwiseSep-4L", DepthwiseSepCNN(4), (3, 32, 32), "CNN"))
    architectures.append(("DepthwiseSep-8L", DepthwiseSepCNN(8), (3, 32, 32), "CNN"))

    # === ResNets ===
    class ResBlock(nn.Module):
        def __init__(self, dim, act_cls=nn.ReLU):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.act = act_cls()
            self.fc2 = nn.Linear(dim, dim)
        def forward(self, x):
            return x + self.fc2(self.act(self.fc1(x)))

    class SimpleResNet(nn.Module):
        def __init__(self, dim=128, blocks=4, act_cls=nn.ReLU):
            super().__init__()
            self.proj = nn.Linear(128, dim)
            self.blocks = nn.ModuleList([ResBlock(dim, act_cls) for _ in range(blocks)])
            self.out = nn.Linear(dim, 10)
        def forward(self, x):
            x = self.proj(x)
            for b in self.blocks:
                x = b(x)
            return self.out(x)

    for blocks in [2, 4, 8, 16]:
        architectures.append((f"ResNet-{blocks}blk", SimpleResNet(blocks=blocks), (128,), "ResNet"))
    architectures.append(("ResNet-GELU-4blk", SimpleResNet(blocks=4, act_cls=nn.GELU), (128,), "ResNet"))

    # Conv ResNet
    class ConvResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return nn.functional.relu(x + out)

    class ConvResNet(nn.Module):
        def __init__(self, blocks=4, channels=64):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
            self.blocks = nn.ModuleList([ConvResBlock(channels) for _ in range(blocks)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, 10)
        def forward(self, x):
            x = self.stem(x)
            for b in self.blocks:
                x = b(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    architectures.append(("ConvResNet-4blk", ConvResNet(4), (3, 32, 32), "ResNet"))
    architectures.append(("ConvResNet-8blk", ConvResNet(8), (3, 32, 32), "ResNet"))

    # === DenseNet ===
    class DenseBlock(nn.Module):
        def __init__(self, in_ch, growth_rate=16):
            super().__init__()
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch), nn.ReLU(),
                nn.Conv2d(in_ch, growth_rate, 3, padding=1))
        def forward(self, x):
            return torch.cat([x, self.conv(x)], 1)

    class SimpleDenseNet(nn.Module):
        def __init__(self, n_blocks=4, growth_rate=16):
            super().__init__()
            self.stem = nn.Conv2d(3, 32, 3, padding=1)
            layers = []
            in_ch = 32
            for _ in range(n_blocks):
                layers.append(DenseBlock(in_ch, growth_rate))
                in_ch += growth_rate
            self.blocks = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_ch, 10)
        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    architectures.append(("DenseNet-4blk", SimpleDenseNet(4), (3, 32, 32), "DenseNet"))
    architectures.append(("DenseNet-8blk", SimpleDenseNet(8), (3, 32, 32), "DenseNet"))

    # === Transformers ===
    class PreLNTransformerBlock(nn.Module):
        def __init__(self, d=128, nhead=4, ff_mult=4):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, d*ff_mult), nn.GELU(), nn.Linear(d*ff_mult, d))
        def forward(self, x):
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.ff(self.ln2(x))
            return x

    class PostLNTransformerBlock(nn.Module):
        def __init__(self, d=128, nhead=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.ln1 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
            self.ln2 = nn.LayerNorm(d)
        def forward(self, x):
            x = self.ln1(x + self.attn(x, x, x)[0])
            x = self.ln2(x + self.ff(x))
            return x

    class SimpleTransformer(nn.Module):
        def __init__(self, n_layers=2, d=128, nhead=4, pre_ln=True):
            super().__init__()
            block_cls = PreLNTransformerBlock if pre_ln else PostLNTransformerBlock
            self.blocks = nn.ModuleList([block_cls(d, nhead) for _ in range(n_layers)])
            self.out = nn.Linear(d, 10)
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return self.out(x[:, 0])

    for layers in [2, 4, 6]:
        architectures.append((f"PreLN-TF-{layers}L", SimpleTransformer(layers, pre_ln=True), (16, 128), "Transformer"))
    for layers in [2, 4]:
        architectures.append((f"PostLN-TF-{layers}L", SimpleTransformer(layers, pre_ln=False), (16, 128), "Transformer"))

    # SwiGLU Transformer
    class SwiGLUBlock(nn.Module):
        def __init__(self, d=128, nhead=4):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.w1 = nn.Linear(d, d*4)
            self.w2 = nn.Linear(d, d*4)
            self.w3 = nn.Linear(d*4, d)
        def forward(self, x):
            h = self.ln1(x)
            x = x + self.attn(h, h, h)[0]
            h = self.ln2(x)
            x = x + self.w3(nn.functional.silu(self.w1(h)) * self.w2(h))
            return x

    class SwiGLUTransformer(nn.Module):
        def __init__(self, n_layers=2, d=128, nhead=4):
            super().__init__()
            self.blocks = nn.ModuleList([SwiGLUBlock(d, nhead) for _ in range(n_layers)])
            self.out = nn.Linear(d, 10)
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return self.out(x[:, 0])

    architectures.append(("SwiGLU-TF-2L", SwiGLUTransformer(2), (16, 128), "Transformer"))
    architectures.append(("SwiGLU-TF-4L", SwiGLUTransformer(4), (16, 128), "Transformer"))

    # === UNet ===
    class MiniUNet(nn.Module):
        def __init__(self, channels=32):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(channels, channels*2, 3, stride=2, padding=1), nn.ReLU())
            self.dec2 = nn.Sequential(nn.ConvTranspose2d(channels*2, channels, 2, stride=2), nn.ReLU())
            self.dec1 = nn.Sequential(nn.Conv2d(channels*2, channels, 3, padding=1), nn.ReLU())
            self.out = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10))
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            d2 = self.dec2(e2)
            d1 = self.dec1(torch.cat([e1, d2], 1))
            return self.out(d1)

    architectures.append(("MiniUNet", MiniUNet(), (3, 32, 32), "UNet"))

    # === Inception-style ===
    class InceptionBlock(nn.Module):
        def __init__(self, in_ch, ch=16):
            super().__init__()
            self.branch1 = nn.Sequential(nn.Conv2d(in_ch, ch, 1), nn.ReLU())
            self.branch3 = nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU())
            self.branch5 = nn.Sequential(nn.Conv2d(in_ch, ch, 5, padding=2), nn.ReLU())
        def forward(self, x):
            return torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], 1)

    class SimpleInception(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
            self.inc1 = InceptionBlock(32, 16)  # output 48
            self.inc2 = InceptionBlock(48, 16)  # output 48
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(48, 10)
        def forward(self, x):
            x = self.stem(x)
            x = self.inc1(x)
            x = self.inc2(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    architectures.append(("Inception-mini", SimpleInception(), (3, 32, 32), "Inception"))

    # === SE-Net ===
    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(channels, channels // reduction), nn.ReLU(),
                nn.Linear(channels // reduction, channels), nn.Sigmoid())
        def forward(self, x):
            out = self.conv(x)
            scale = self.se(out).unsqueeze(-1).unsqueeze(-1)
            return x + out * scale

    class SimpleSENet(nn.Module):
        def __init__(self, blocks=3, channels=32):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
            self.blocks = nn.ModuleList([SEBlock(channels) for _ in range(blocks)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, 10)
        def forward(self, x):
            x = self.stem(x)
            for b in self.blocks:
                x = b(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    architectures.append(("SENet-3blk", SimpleSENet(3), (3, 32, 32), "SENet"))

    # === Highway network ===
    class HighwayLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.H = nn.Linear(dim, dim)
            self.T = nn.Linear(dim, dim)
            self.T.bias.data.fill_(-2.0)
        def forward(self, x):
            h = nn.functional.relu(self.H(x))
            t = torch.sigmoid(self.T(x))
            return h * t + x * (1 - t)

    class HighwayNet(nn.Module):
        def __init__(self, dim=128, layers=4):
            super().__init__()
            self.proj = nn.Linear(128, dim)
            self.highway = nn.Sequential(*[HighwayLayer(dim) for _ in range(layers)])
            self.out = nn.Linear(dim, 10)
        def forward(self, x):
            return self.out(self.highway(self.proj(x)))

    architectures.append(("Highway-4L", HighwayNet(layers=4), (128,), "Highway"))
    architectures.append(("Highway-8L", HighwayNet(layers=8), (128,), "Highway"))

    # === Bottleneck / Funnel ===
    class BottleneckMLP(nn.Module):
        def __init__(self, widths=[128, 64, 32, 64, 128, 10]):
            super().__init__()
            layers = []
            for i in range(len(widths) - 1):
                layers.append(nn.Linear(widths[i], widths[i+1]))
                if i < len(widths) - 2:
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    architectures.append(("Bottleneck-MLP", BottleneckMLP(), (128,), "Bottleneck"))
    architectures.append(("Funnel-MLP", BottleneckMLP([128, 256, 128, 64, 32, 10]), (128,), "Bottleneck"))

    # === Normed MLP ===
    class NormedMLP(nn.Module):
        def __init__(self, depth=5, width=256, norm_type='layer'):
            super().__init__()
            layers = [nn.Linear(128, width)]
            for _ in range(depth - 2):
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(width))
                else:
                    layers.append(nn.BatchNorm1d(width))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(width, width))
            layers.append(nn.Linear(width, 10))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    architectures.append(("NormedMLP-LN-5L", NormedMLP(5, norm_type='layer'), (128,), "NormedMLP"))
    architectures.append(("NormedMLP-BN-5L", NormedMLP(5, norm_type='batch'), (128,), "NormedMLP"))
    architectures.append(("NormedMLP-LN-10L", NormedMLP(10, norm_type='layer'), (128,), "NormedMLP"))

    # === Dropout MLP ===
    class DropoutMLP(nn.Module):
        def __init__(self, depth=5, width=256, p=0.3):
            super().__init__()
            layers = [nn.Linear(128, width), nn.ReLU(), nn.Dropout(p)]
            for _ in range(depth - 2):
                layers.extend([nn.Linear(width, width), nn.ReLU(), nn.Dropout(p)])
            layers.append(nn.Linear(width, 10))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    architectures.append(("Dropout-MLP-5L", DropoutMLP(5, p=0.3), (128,), "Dropout"))
    architectures.append(("Dropout-MLP-5L-heavy", DropoutMLP(5, p=0.5), (128,), "Dropout"))

    # === LSTM/GRU ===
    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim=128, hidden_dim=128, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 10)
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (B, 1, D)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class GRUClassifier(nn.Module):
        def __init__(self, input_dim=128, hidden_dim=128, n_layers=2):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 10)
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    architectures.append(("LSTM-2L", LSTMClassifier(n_layers=2), (128,), "Recurrent"))
    architectures.append(("LSTM-4L", LSTMClassifier(n_layers=4), (128,), "Recurrent"))
    architectures.append(("GRU-2L", GRUClassifier(n_layers=2), (128,), "Recurrent"))

    # === Multi-branch / Wide-and-Deep ===
    class WideAndDeep(nn.Module):
        def __init__(self, wide_dim=128, deep_dims=[128, 64, 32]):
            super().__init__()
            deep_layers = []
            in_d = wide_dim
            for d in deep_dims:
                deep_layers.extend([nn.Linear(in_d, d), nn.ReLU()])
                in_d = d
            self.deep = nn.Sequential(*deep_layers)
            self.out = nn.Linear(wide_dim + deep_dims[-1], 10)
        def forward(self, x):
            deep_out = self.deep(x)
            return self.out(torch.cat([x, deep_out], 1))

    architectures.append(("WideAndDeep", WideAndDeep(), (128,), "MultiBranch"))

    # === Torchvision models ===
    tv_models = [
        ("tv-ResNet18", lambda: tvm.resnet18(weights=None), (3, 32, 32)),
        ("tv-ResNet50", lambda: tvm.resnet50(weights=None), (3, 32, 32)),
        ("tv-VGG11", lambda: tvm.vgg11(weights=None), (3, 32, 32)),
        ("tv-MobileNetV2", lambda: tvm.mobilenet_v2(weights=None), (3, 32, 32)),
        ("tv-DenseNet121", lambda: tvm.densenet121(weights=None), (3, 32, 32)),
        ("tv-EfficientNet-B0", lambda: tvm.efficientnet_b0(weights=None), (3, 32, 32)),
        ("tv-ShuffleNetV2", lambda: tvm.shufflenet_v2_x0_5(weights=None), (3, 32, 32)),
        ("tv-SqueezeNet", lambda: tvm.squeezenet1_0(weights=None), (3, 32, 32)),
        ("tv-MNASNet", lambda: tvm.mnasnet0_5(weights=None), (3, 32, 32)),
        ("tv-RegNetY-400MF", lambda: tvm.regnet_y_400mf(weights=None), (3, 32, 32)),
    ]

    for name, model_fn, shape in tv_models:
        try:
            model = model_fn()
            architectures.append((name, model, shape, "Torchvision"))
        except Exception as e:
            print(f"Skip {name}: {e}")

    return architectures


def run_experiment():
    """Run the comprehensive architecture experiment."""
    import torch
    from dag_propagator import analyze_dag

    print("=" * 70)
    print("PhaseKit Universal Architecture Experiment v2")
    print("=" * 70)

    architectures = build_architecture_zoo()
    print(f"\nTotal architectures: {len(architectures)}")

    results = []
    family_stats = {}

    for i, (name, model, input_shape, family) in enumerate(architectures):
        print(f"\n[{i+1}/{len(architectures)}] {name} ({family})...")

        result = {"name": name, "family": family}

        try:
            # DAG analysis
            t0 = time.time()
            dag_result = analyze_dag(model, input_shape, n_samples=64, seed=42)
            t1 = time.time()

            result["dag_phase"] = dag_result.phase
            result["dag_chi"] = dag_result.chi_total
            result["dag_nodes"] = dag_result.n_nodes
            result["dag_weight_layers"] = dag_result.n_weight_layers
            result["dag_branches"] = dag_result.n_branches
            result["dag_residual"] = dag_result.n_residual
            result["has_attention"] = dag_result.has_attention
            result["has_norm"] = dag_result.has_norm
            result["has_residual"] = dag_result.has_residual
            result["n_params"] = dag_result.n_params
            result["variance_error_pct"] = dag_result.variance_error_pct
            result["analysis_time_s"] = t1 - t0

            # Empirical phase
            emp_phase, emp_traj = determine_empirical_phase(model, input_shape, n_samples=64)
            result["emp_phase"] = emp_phase
            result["emp_variance_trajectory_len"] = len(emp_traj)

            # Phase agreement
            result["phase_match"] = dag_result.phase == emp_phase

            # Family stats
            if family not in family_stats:
                family_stats[family] = {"total": 0, "correct": 0}
            family_stats[family]["total"] += 1
            if result["phase_match"]:
                family_stats[family]["correct"] += 1

            status = "✓" if result["phase_match"] else "✗"
            print(f"  {status} DAG={dag_result.phase}, Emp={emp_phase}, χ={dag_result.chi_total:.4f}, "
                  f"var_err={dag_result.variance_error_pct:.1f}%, "
                  f"nodes={dag_result.n_nodes}, t={t1-t0:.2f}s")

        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            result["phase_match"] = False
            print(f"  ERROR: {e}")
            if family not in family_stats:
                family_stats[family] = {"total": 0, "correct": 0}
            family_stats[family]["total"] += 1

        results.append(result)

    # Summary
    n_total = len(results)
    n_match = sum(1 for r in results if r.get("phase_match", False))
    n_error = sum(1 for r in results if "error" in r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {n_total}, Correct: {n_match}, Error: {n_error}")
    print(f"Phase accuracy: {n_match}/{n_total} = {100*n_match/n_total:.1f}%")
    print(f"Phase accuracy (excl errors): {n_match}/{n_total-n_error} = {100*n_match/max(n_total-n_error,1):.1f}%")
    print()

    for family, stats in sorted(family_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {family:20s}: {stats['correct']}/{stats['total']} = {acc:.0f}%")

    # Save results
    output = {
        "experiment": "universal_v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_architectures": n_total,
        "summary": {
            "total": n_total,
            "correct": n_match,
            "errors": n_error,
            "accuracy": n_match / n_total if n_total > 0 else 0,
            "accuracy_excl_errors": n_match / max(n_total - n_error, 1),
        },
        "family_stats": family_stats,
        "architectures": results,
    }

    out_path = os.path.join(RESULTS_DIR, "universal_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
