"""
Developer workflow validation: prove PhaseKit is useful for real PyTorch developers.

Tests the core claim: "PhaseKit's check() correctly predicts training
success/failure, and fix() improves convergence on real architectures."

Architecture zoo includes:
- MLPs (2-50 layers, ReLU/GELU/SiLU/Tanh, good/bad init)
- CNNs (plain, VGG-style, with/without BatchNorm)
- ResNets (with skip connections)
- Transformers (GPT-2, BERT, ViT, T5-style)
- RNNs (LSTM, GRU)
- Modern architectures (ConvNeXt-style, MLP-Mixer)
- HuggingFace configs (GPT2, BERT, DistilBERT, ViT, Whisper, LLaMA-style)

Protocol:
1. Define 50+ diverse nn.Module architectures
2. For each: run check() to predict trainability
3. Train each model on a simple task for N steps
4. Measure final loss and convergence speed
5. For models where check() says "fix needed", apply fix() and re-train
6. Compute prediction accuracy and fix() effectiveness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def make_dataset(input_shape, n_samples=128, n_classes=10, seed=42):
    """Create a simple classification dataset."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, *input_shape)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def train_model(model, X, y, n_steps=200, lr=0.001, seed=42):
    """Train a model and return per-step losses."""
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        try:
            out = model(X)
            if out.dim() == 3:
                out = out[:, -1, :]  # take last token for seq models
            if out.dim() > 2:
                out = out.view(out.shape[0], -1)
                out = out[:, :max(y.max().item() + 1, 10)]
            if out.shape[-1] < y.max() + 1:
                out = nn.functional.pad(out, (0, int(y.max() + 1 - out.shape[-1])))
            loss = criterion(out, y)
            if not torch.isfinite(loss):
                losses.append(float('inf'))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        except Exception as e:
            losses.append(float('inf'))
            break
    return losses


# ──────────────────────────────────────────────────────────────
# Architecture zoo
# ──────────────────────────────────────────────────────────────

def make_bad_init_mlp(depth, width, sigma_w, act_cls):
    layers = []
    for i in range(depth - 1):
        lin = nn.Linear(width, width)
        nn.init.normal_(lin.weight, 0, sigma_w / width**0.5)
        nn.init.zeros_(lin.bias)
        layers.extend([lin, act_cls()])
    out = nn.Linear(width, 10)
    nn.init.normal_(out.weight, 0, 0.01)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim, act_cls=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = act_cls()
    def forward(self, x):
        return x + self.act(self.fc2(self.act(self.fc1(x))))


class SimpleResNet(nn.Module):
    def __init__(self, dim, n_blocks, act_cls=nn.ReLU):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(dim, act_cls) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 10)
    def forward(self, x):
        return self.head(self.blocks(x))


class ResBlockLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.fc2(self.act(self.ln(self.fc1(x)))))


class ResNetLN(nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlockLN(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 10)
    def forward(self, x):
        return self.head(self.blocks(x))


class MiniTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 10)
    def forward(self, x):
        return self.head(self.enc(x).mean(dim=1))


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 10)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 10)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise conv + LayerNorm + pointwise."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)
    def forward(self, x):
        res = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return res + x


class MiniConvNeXt(nn.Module):
    def __init__(self, dim=32, n_blocks=3):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, 4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, 10))
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class MLPMixerBlock(nn.Module):
    """MLP-Mixer block: token mixing + channel mixing."""
    def __init__(self, n_patches, dim, token_dim, channel_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(nn.Linear(n_patches, token_dim), nn.GELU(), nn.Linear(token_dim, n_patches))
        self.ln2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(nn.Linear(dim, channel_dim), nn.GELU(), nn.Linear(channel_dim, dim))
    def forward(self, x):
        x = x + self.token_mix(self.ln1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(self.ln2(x))
        return x


class MiniMLPMixer(nn.Module):
    def __init__(self, n_patches=4, dim=64, n_blocks=2):
        super().__init__()
        self.proj = nn.Linear(48, dim)  # 3*4*4 patch
        self.blocks = nn.Sequential(*[MLPMixerBlock(n_patches, dim, dim//2, dim*2) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
    def forward(self, x):
        B = x.shape[0]
        # Patchify: (B,3,8,8) -> (B,4,48)
        x = x.reshape(B, 3, 2, 4, 2, 4).permute(0, 2, 4, 1, 3, 5).reshape(B, 4, 48)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.ln(x.mean(dim=1))
        return self.head(x)


class VGGBlock(nn.Module):
    """VGG-style block: conv-conv-pool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)


class MiniVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(VGGBlock(3, 16), VGGBlock(16, 32))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 2 * 2, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.classifier(self.features(x))


def make_hf_architectures():
    """Create architectures from HuggingFace model configs."""
    archs = []

    # --- GPT-2 variants ---
    try:
        from transformers import GPT2Config, GPT2Model
        # GPT-2 Tiny (for speed)
        cfg = GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=100, n_positions=32)
        class GPT2Classifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.gpt2 = GPT2Model(cfg)
                self.head = nn.Linear(cfg.n_embd, 10)
            def forward(self, x):
                # x is (B, seq_len, n_embd) — we use as inputs_embeds
                out = self.gpt2(inputs_embeds=x).last_hidden_state[:, -1]
                return self.head(out)
        archs.append(("HF-GPT2-2L", GPT2Classifier(cfg), (16, 64)))

        cfg4 = GPT2Config(n_embd=64, n_layer=4, n_head=4, vocab_size=100, n_positions=32)
        archs.append(("HF-GPT2-4L", GPT2Classifier(cfg4), (16, 64)))

        cfg6 = GPT2Config(n_embd=128, n_layer=6, n_head=4, vocab_size=100, n_positions=32)
        archs.append(("HF-GPT2-6L-128d", GPT2Classifier(cfg6), (16, 128)))

        # GPT-2 Small config (12 layers, 768 dim) — reduced seq length for speed
        cfg_sm = GPT2Config(n_embd=128, n_layer=12, n_head=4, vocab_size=100, n_positions=32)
        archs.append(("HF-GPT2-12L-128d", GPT2Classifier(cfg_sm), (8, 128)))
    except Exception as e:
        print(f"  [skip GPT2: {e}]")

    # --- BERT variants ---
    try:
        from transformers import BertConfig, BertModel
        class BERTClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.bert = BertModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                out = self.bert(inputs_embeds=x).last_hidden_state[:, 0]
                return self.head(out)

        cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=128, vocab_size=100, max_position_embeddings=32)
        archs.append(("HF-BERT-2L", BERTClassifier(cfg), (16, 64)))

        cfg6 = BertConfig(hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
                          intermediate_size=256, vocab_size=100, max_position_embeddings=32)
        archs.append(("HF-BERT-6L-128d", BERTClassifier(cfg6), (16, 128)))

        cfg12 = BertConfig(hidden_size=128, num_hidden_layers=12, num_attention_heads=4,
                           intermediate_size=256, vocab_size=100, max_position_embeddings=32)
        archs.append(("HF-BERT-12L-128d", BERTClassifier(cfg12), (8, 128)))
    except Exception as e:
        print(f"  [skip BERT: {e}]")

    # --- DistilBERT ---
    try:
        from transformers import DistilBertConfig, DistilBertModel
        class DistilBERTClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = DistilBertModel(cfg)
                self.head = nn.Linear(cfg.dim, 10)
            def forward(self, x):
                out = self.model(inputs_embeds=x).last_hidden_state[:, 0]
                return self.head(out)

        cfg = DistilBertConfig(dim=64, n_layers=3, n_heads=4,
                               hidden_dim=128, vocab_size=100, max_position_embeddings=32)
        archs.append(("HF-DistilBERT-3L", DistilBERTClassifier(cfg), (16, 64)))
    except Exception as e:
        print(f"  [skip DistilBERT: {e}]")

    # --- ViT ---
    try:
        from transformers import ViTConfig, ViTModel
        class ViTClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.vit = ViTModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                # x: (B, 3, image_size, image_size)
                out = self.vit(pixel_values=x).last_hidden_state[:, 0]
                return self.head(out)

        cfg = ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=128, image_size=32, patch_size=8, num_channels=3)
        archs.append(("HF-ViT-2L", ViTClassifier(cfg), (3, 32, 32)))

        cfg4 = ViTConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                         intermediate_size=128, image_size=32, patch_size=8, num_channels=3)
        archs.append(("HF-ViT-4L", ViTClassifier(cfg4), (3, 32, 32)))

        cfg6 = ViTConfig(hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
                         intermediate_size=256, image_size=32, patch_size=8, num_channels=3)
        archs.append(("HF-ViT-6L-128d", ViTClassifier(cfg6), (3, 32, 32)))
    except Exception as e:
        print(f"  [skip ViT: {e}]")

    # --- Whisper encoder ---
    try:
        from transformers import WhisperConfig, WhisperModel
        class WhisperEncoder(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = WhisperModel(cfg)
                self.head = nn.Linear(cfg.d_model, 10)
            def forward(self, x):
                # x: (B, n_mels, seq_len) — treat as mel spectrogram
                out = self.model.encoder(x).last_hidden_state.mean(dim=1)
                return self.head(out)

        cfg = WhisperConfig(d_model=64, encoder_layers=2, decoder_layers=1,
                            encoder_attention_heads=4, decoder_attention_heads=4,
                            encoder_ffn_dim=128, decoder_ffn_dim=128,
                            vocab_size=100, num_mel_bins=32, max_source_positions=16)
        archs.append(("HF-Whisper-2L", WhisperEncoder(cfg), (32, 16)))
    except Exception as e:
        print(f"  [skip Whisper: {e}]")

    # --- LLaMA-style (RoPE, RMSNorm, SwiGLU) ---
    try:
        from transformers import LlamaConfig, LlamaModel
        class LlamaClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = LlamaModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                out = self.model(inputs_embeds=x).last_hidden_state[:, -1]
                return self.head(out)

        cfg = LlamaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=128, vocab_size=100, max_position_embeddings=32,
                          num_key_value_heads=2)
        archs.append(("HF-LLaMA-2L", LlamaClassifier(cfg), (16, 64)))

        cfg4 = LlamaConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                           intermediate_size=128, vocab_size=100, max_position_embeddings=32,
                           num_key_value_heads=2)
        archs.append(("HF-LLaMA-4L", LlamaClassifier(cfg4), (16, 64)))

        cfg8 = LlamaConfig(hidden_size=128, num_hidden_layers=8, num_attention_heads=4,
                           intermediate_size=256, vocab_size=100, max_position_embeddings=32,
                           num_key_value_heads=2)
        archs.append(("HF-LLaMA-8L-128d", LlamaClassifier(cfg8), (8, 128)))
    except Exception as e:
        print(f"  [skip LLaMA: {e}]")

    # --- Mistral ---
    try:
        from transformers import MistralConfig, MistralModel
        class MistralClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = MistralModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                out = self.model(inputs_embeds=x).last_hidden_state[:, -1]
                return self.head(out)

        cfg = MistralConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=128, vocab_size=100, max_position_embeddings=32,
                            num_key_value_heads=2, sliding_window=16)
        archs.append(("HF-Mistral-2L", MistralClassifier(cfg), (16, 64)))

        cfg4 = MistralConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                             intermediate_size=256, vocab_size=100, max_position_embeddings=32,
                             num_key_value_heads=2, sliding_window=16)
        archs.append(("HF-Mistral-4L-128d", MistralClassifier(cfg4), (16, 128)))
    except Exception as e:
        print(f"  [skip Mistral: {e}]")

    # --- T5 (encoder-decoder) ---
    try:
        from transformers import T5Config, T5EncoderModel
        class T5Classifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = T5EncoderModel(cfg)
                self.head = nn.Linear(cfg.d_model, 10)
            def forward(self, x):
                out = self.model(inputs_embeds=x).last_hidden_state.mean(dim=1)
                return self.head(out)

        cfg = T5Config(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                       vocab_size=100, d_kv=16)
        archs.append(("HF-T5Enc-2L", T5Classifier(cfg), (16, 64)))
    except Exception as e:
        print(f"  [skip T5: {e}]")

    # --- Electra ---
    try:
        from transformers import ElectraConfig, ElectraModel
        class ElectraClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = ElectraModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                out = self.model(inputs_embeds=x).last_hidden_state[:, 0]
                return self.head(out)

        cfg = ElectraConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=128, vocab_size=100, max_position_embeddings=32,
                            embedding_size=64)
        archs.append(("HF-Electra-2L", ElectraClassifier(cfg), (16, 64)))
    except Exception as e:
        print(f"  [skip Electra: {e}]")

    # --- RoBERTa (same as BERT but different config defaults) ---
    try:
        from transformers import RobertaConfig, RobertaModel
        class RoBERTaClassifier(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = RobertaModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                out = self.model(inputs_embeds=x).last_hidden_state[:, 0]
                return self.head(out)

        cfg = RobertaConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                            intermediate_size=128, vocab_size=100, max_position_embeddings=34,
                            type_vocab_size=1)
        archs.append(("HF-RoBERTa-4L", RoBERTaClassifier(cfg), (16, 64)))
    except Exception as e:
        print(f"  [skip RoBERTa: {e}]")

    return archs


def make_architectures():
    """Return list of (name, model, input_shape) tuples."""
    archs = []

    # ═══════ MLPs ═══════
    archs.append(("MLP-2L-ReLU", nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)), (64,)))
    archs.append(("MLP-2L-GELU", nn.Sequential(
        nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 10)), (64,)))
    archs.append(("MLP-5L-ReLU", nn.Sequential(
        *[nn.Sequential(nn.Linear(128, 128), nn.ReLU()) for _ in range(4)],
        nn.Linear(128, 10)), (128,)))
    archs.append(("MLP-5L-GELU", nn.Sequential(
        *[nn.Sequential(nn.Linear(128, 128), nn.GELU()) for _ in range(4)],
        nn.Linear(128, 10)), (128,)))
    archs.append(("MLP-5L-SiLU", nn.Sequential(
        *[nn.Sequential(nn.Linear(128, 128), nn.SiLU()) for _ in range(4)],
        nn.Linear(128, 10)), (128,)))
    archs.append(("MLP-5L-Tanh", nn.Sequential(
        *[nn.Sequential(nn.Linear(128, 128), nn.Tanh()) for _ in range(4)],
        nn.Linear(128, 10)), (128,)))
    archs.append(("MLP-10L-ReLU", nn.Sequential(
        *[nn.Sequential(nn.Linear(128, 128), nn.ReLU()) for _ in range(9)],
        nn.Linear(128, 10)), (128,)))
    archs.append(("MLP-20L-ReLU", nn.Sequential(
        *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(19)],
        nn.Linear(64, 10)), (64,)))
    archs.append(("MLP-20L-GELU", nn.Sequential(
        *[nn.Sequential(nn.Linear(64, 64), nn.GELU()) for _ in range(19)],
        nn.Linear(64, 10)), (64,)))
    archs.append(("MLP-30L-ReLU", nn.Sequential(
        *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(29)],
        nn.Linear(64, 10)), (64,)))

    # Bad init MLPs
    archs.append(("MLP-5L-ReLU-high-sw", make_bad_init_mlp(5, 128, 3.0, nn.ReLU), (128,)))
    archs.append(("MLP-5L-ReLU-low-sw", make_bad_init_mlp(5, 128, 0.3, nn.ReLU), (128,)))
    archs.append(("MLP-10L-GELU-low-sw", make_bad_init_mlp(10, 128, 0.3, nn.GELU), (128,)))

    # MLP with LayerNorm
    archs.append(("MLP-10L-LN-GELU", nn.Sequential(
        *[nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU()) for _ in range(9)],
        nn.Linear(128, 10)), (128,)))
    archs.append(("MLP-20L-LN-GELU", nn.Sequential(
        *[nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU()) for _ in range(19)],
        nn.Linear(64, 10)), (64,)))

    # ═══════ CNNs ═══════
    archs.append(("CNN-3L", nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10)), (3, 8, 8)))
    archs.append(("CNN-5L-BN", nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10)), (3, 8, 8)))
    archs.append(("MiniVGG", MiniVGG(), (3, 8, 8)))

    # ═══════ ResNets ═══════
    archs.append(("ResNet-5B-ReLU", SimpleResNet(128, 5, nn.ReLU), (128,)))
    archs.append(("ResNet-10B-ReLU", SimpleResNet(128, 10, nn.ReLU), (128,)))
    archs.append(("ResNet-5B-GELU", SimpleResNet(128, 5, nn.GELU), (128,)))
    archs.append(("ResNet-5B-LN", ResNetLN(128, 5), (128,)))

    # ═══════ Transformers (PyTorch native) ═══════
    archs.append(("Transformer-2L", MiniTransformer(64, 4, 2), (8, 64)))
    archs.append(("Transformer-4L", MiniTransformer(64, 4, 4), (8, 64)))
    archs.append(("Transformer-6L", MiniTransformer(64, 4, 6), (8, 64)))

    # ═══════ RNNs ═══════
    archs.append(("LSTM-2L", LSTMModel(64, 128, 2), (8, 64)))
    archs.append(("GRU-2L", GRUModel(64, 128, 2), (8, 64)))
    archs.append(("LSTM-4L", LSTMModel(64, 128, 4), (8, 64)))

    # ═══════ Modern architectures ═══════
    archs.append(("ConvNeXt-3B", MiniConvNeXt(32, 3), (3, 16, 16)))
    archs.append(("MLPMixer-2B", MiniMLPMixer(4, 64, 2), (3, 8, 8)))

    # ═══════ HuggingFace models ═══════
    hf_archs = make_hf_architectures()
    archs.extend(hf_archs)

    return archs


def run_experiment():
    """Run the full developer workflow validation."""
    from developer_api import check, fix

    archs = make_architectures()
    print(f"Testing {len(archs)} architectures...")

    results = []

    for i, (name, model, input_shape) in enumerate(archs):
        print(f"\n[{i+1}/{len(archs)}] {name}")

        # 1. PhaseKit check
        report = check(model, input_shape, verbose=False)

        # 2. Train with default init (fewer samples for HF models)
        n_samp = 64 if name.startswith("HF-") else 128
        X, y = make_dataset(input_shape, n_samples=n_samp, seed=42)
        default_losses = train_model(model, X, y, n_steps=200, lr=0.001, seed=42)
        default_final = default_losses[-1] if default_losses else float('inf')

        # 3. If fix needed, apply fix and re-train
        fixed_final = None
        fix_improvement = None
        if report.fix_needed:
            # Re-create a fresh copy of the same architecture for fair comparison
            arch_idx = i
            _, model_fresh, _ = make_architectures()[arch_idx]
            try:
                fix_report = fix(model_fresh, input_shape, verbose=False)
                fixed_losses = train_model(model_fresh, X, y, n_steps=200, lr=0.001, seed=42)
                fixed_final = fixed_losses[-1] if fixed_losses else float('inf')
                if np.isfinite(default_final) and np.isfinite(fixed_final) and default_final > 0:
                    fix_improvement = default_final / max(fixed_final, 1e-10)
                elif np.isfinite(fixed_final) and not np.isfinite(default_final):
                    fix_improvement = float('inf')  # fixed a divergence
            except Exception as e:
                fixed_final = float('inf')

        # Determine if model actually trained well
        # For CNNs on random data, convergence is slower — use relative improvement
        # For all: if loss decreased significantly from initial (~2.3 for 10-class),
        # the model is training (init is working)
        initial_loss = default_losses[0] if default_losses else float('inf')
        loss_reduction = (initial_loss - default_final) / max(initial_loss, 1e-10) if np.isfinite(initial_loss) and np.isfinite(default_final) else 0
        trained_well = np.isfinite(default_final) and (
            default_final < 0.5 or  # absolute threshold
            loss_reduction > 0.5     # or >50% reduction from start
        )

        # Convergence speed: steps to reach loss < 1.0
        steps_to_converge = len(default_losses)
        for s, l in enumerate(default_losses):
            if l < 1.0:
                steps_to_converge = s
                break

        result = {
            "name": name,
            "depth": report.depth,
            "width": report.width,
            "activation": report.activation,
            "arch_type": report.arch_type,
            "n_params": report.n_params,
            "check_healthy": report.healthy,
            "check_phase": report.phase,
            "check_confidence": report.confidence,
            "check_variance_health": report.variance_health,
            "actually_trained": trained_well,
            "default_final_loss": default_final if np.isfinite(default_final) else "inf",
            "steps_to_converge": steps_to_converge,
            "fix_needed": report.fix_needed,
            "fixed_final_loss": fixed_final if fixed_final is not None and np.isfinite(fixed_final) else ("inf" if fixed_final is not None else None),
            "fix_improvement": f"{fix_improvement:.1f}x" if fix_improvement is not None and np.isfinite(fix_improvement) else None,
        }
        results.append(result)

        # Print summary
        status = "✓" if trained_well == report.healthy else "✗"
        loss_str = f"{default_final:.3f}" if np.isfinite(default_final) else "inf"
        print(f"  {status} check={report.phase}({'healthy' if report.healthy else 'fix needed'}) "
              f"actual={'trained' if trained_well else 'failed'} loss={loss_str}", end="")
        if fixed_final is not None:
            fix_str = f"{fixed_final:.3f}" if np.isfinite(fixed_final) else "inf"
            print(f" → fixed={fix_str}", end="")
            if fix_improvement is not None and np.isfinite(fix_improvement):
                print(f" ({fix_improvement:.1f}× better)", end="")
        print()

    # Compute metrics
    n_correct = sum(1 for r in results if r["check_healthy"] == r["actually_trained"])
    n_total = len(results)
    accuracy = n_correct / n_total

    # Precision/recall for "fix needed" prediction
    tp = sum(1 for r in results if r["fix_needed"] and not r["actually_trained"])
    fp = sum(1 for r in results if r["fix_needed"] and r["actually_trained"])
    fn = sum(1 for r in results if not r["fix_needed"] and not r["actually_trained"])
    tn = sum(1 for r in results if not r["fix_needed"] and r["actually_trained"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # Fix improvement stats
    fix_results = [r for r in results if r["fix_improvement"] is not None]
    fix_improvements = []
    for r in fix_results:
        try:
            v = float(r["fix_improvement"].replace("x", ""))
            if np.isfinite(v):
                fix_improvements.append(v)
        except (ValueError, AttributeError):
            pass

    mean_fix_improvement = np.mean(fix_improvements) if fix_improvements else 0
    n_fix_helped = sum(1 for f in fix_improvements if f > 1.0)

    print(f"\n{'='*70}")
    print(f"DEVELOPER WORKFLOW VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Architectures tested: {n_total}")
    print(f"Prediction accuracy:  {accuracy:.0%} ({n_correct}/{n_total})")
    print(f"  Precision (fix needed):  {precision:.0%} (TP={tp}, FP={fp})")
    print(f"  Recall (fix needed):     {recall:.0%} (TP={tp}, FN={fn})")
    print(f"fix() applied to:     {len(fix_results)} models")
    print(f"  fix() helped:       {n_fix_helped}/{len(fix_results)}")
    if fix_improvements:
        print(f"  Mean improvement:   {mean_fix_improvement:.1f}×")
        print(f"  Median improvement: {np.median(fix_improvements):.1f}×")
    print(f"{'='*70}")

    # Save results
    summary = {
        "n_architectures": n_total,
        "prediction_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "n_fix_applied": len(fix_results),
        "n_fix_helped": n_fix_helped,
        "mean_fix_improvement": mean_fix_improvement,
        "median_fix_improvement": float(np.median(fix_improvements)) if fix_improvements else 0,
        "details": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'developer_validation')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'developer_validation_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_experiment()
