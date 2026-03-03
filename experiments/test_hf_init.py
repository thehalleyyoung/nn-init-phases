"""
Test and benchmark PhaseKit init_hf() on HuggingFace model configs.

For each config:
1. Create model with HF default init → train → measure loss
2. Create model with PhaseKit init → train → measure loss
3. Compare convergence speed and final loss

Saves all results to experiments/results/hf_init_benchmark/hf_init_results.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_data(input_shape, n_samples=64, n_classes=10, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, *input_shape)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def train(model, X, y, n_steps=300, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        try:
            out = model(X)
            if out.dim() == 3:
                out = out[:, -1, :]
            if out.dim() > 2:
                out = out.view(out.shape[0], -1)[:, :max(y.max().item()+1, 10)]
            if out.shape[-1] < y.max() + 1:
                out = nn.functional.pad(out, (0, int(y.max()+1 - out.shape[-1])))
            loss = crit(out, y)
            if not torch.isfinite(loss):
                losses.append(float('inf'))
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        except Exception:
            losses.append(float('inf'))
            break
    return losses


def steps_to_threshold(losses, threshold=1.0):
    for i, l in enumerate(losses):
        if l < threshold:
            return i
    return len(losses)


# ── HuggingFace model builders ──────────────────────────────

def build_hf_configs():
    """Return list of (name, config, model_builder, input_shape)."""
    configs = []

    # GPT-2 variants
    try:
        from transformers import GPT2Config, GPT2Model
        class GPT2Clf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.gpt2 = GPT2Model(cfg)
                self.head = nn.Linear(cfg.n_embd, 10)
            def forward(self, x):
                return self.head(self.gpt2(inputs_embeds=x).last_hidden_state[:, -1])

        for n_layer, n_embd, label in [(2, 64, '2L-64d'), (4, 64, '4L-64d'),
                                        (6, 128, '6L-128d'), (12, 128, '12L-128d')]:
            cfg = GPT2Config(n_embd=n_embd, n_layer=n_layer, n_head=max(4, n_embd//16),
                             vocab_size=100, n_positions=32)
            configs.append((f'GPT2-{label}', cfg,
                            lambda c=cfg: GPT2Clf(c), (16, n_embd)))
    except Exception as e:
        print(f'[skip GPT2: {e}]')

    # BERT variants
    try:
        from transformers import BertConfig, BertModel
        class BERTClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.bert = BertModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                return self.head(self.bert(inputs_embeds=x).last_hidden_state[:, 0])

        for n_layer, hidden, label in [(2, 64, '2L-64d'), (6, 128, '6L-128d'),
                                        (12, 128, '12L-128d')]:
            cfg = BertConfig(hidden_size=hidden, num_hidden_layers=n_layer,
                             num_attention_heads=max(4, hidden//16),
                             intermediate_size=hidden*4, vocab_size=100,
                             max_position_embeddings=34)
            configs.append((f'BERT-{label}', cfg,
                            lambda c=cfg: BERTClf(c), (16, hidden)))
    except Exception as e:
        print(f'[skip BERT: {e}]')

    # ViT
    try:
        from transformers import ViTConfig, ViTModel
        class ViTClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.vit = ViTModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                return self.head(self.vit(pixel_values=x).last_hidden_state[:, 0])

        for n_layer, hidden, label in [(2, 64, '2L-64d'), (4, 64, '4L-64d'),
                                        (6, 128, '6L-128d')]:
            cfg = ViTConfig(hidden_size=hidden, num_hidden_layers=n_layer,
                            num_attention_heads=max(4, hidden//16),
                            intermediate_size=hidden*4, image_size=32,
                            patch_size=8, num_channels=3)
            configs.append((f'ViT-{label}', cfg,
                            lambda c=cfg: ViTClf(c), (3, 32, 32)))
    except Exception as e:
        print(f'[skip ViT: {e}]')

    # LLaMA
    try:
        from transformers import LlamaConfig, LlamaModel
        class LlamaClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = LlamaModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                return self.head(self.model(inputs_embeds=x).last_hidden_state[:, -1])

        for n_layer, hidden, label in [(2, 64, '2L-64d'), (4, 64, '4L-64d')]:
            cfg = LlamaConfig(hidden_size=hidden, num_hidden_layers=n_layer,
                              num_attention_heads=max(4, hidden//16),
                              intermediate_size=hidden*4, vocab_size=100,
                              max_position_embeddings=32,
                              num_key_value_heads=max(2, hidden//32))
            configs.append((f'LLaMA-{label}', cfg,
                            lambda c=cfg: LlamaClf(c), (16, hidden)))
    except Exception as e:
        print(f'[skip LLaMA: {e}]')

    # Mistral
    try:
        from transformers import MistralConfig, MistralModel
        class MistralClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = MistralModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                return self.head(self.model(inputs_embeds=x).last_hidden_state[:, -1])

        for n_layer, hidden, label in [(2, 64, '2L-64d'), (4, 128, '4L-128d')]:
            cfg = MistralConfig(hidden_size=hidden, num_hidden_layers=n_layer,
                                num_attention_heads=max(4, hidden//16),
                                intermediate_size=hidden*4, vocab_size=100,
                                max_position_embeddings=32,
                                num_key_value_heads=max(2, hidden//32),
                                sliding_window=16)
            configs.append((f'Mistral-{label}', cfg,
                            lambda c=cfg: MistralClf(c), (16, hidden)))
    except Exception as e:
        print(f'[skip Mistral: {e}]')

    # DistilBERT
    try:
        from transformers import DistilBertConfig, DistilBertModel
        class DistilClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = DistilBertModel(cfg)
                self.head = nn.Linear(cfg.dim, 10)
            def forward(self, x):
                return self.head(self.model(inputs_embeds=x).last_hidden_state[:, 0])

        cfg = DistilBertConfig(dim=64, n_layers=3, n_heads=4,
                               hidden_dim=128, vocab_size=100,
                               max_position_embeddings=32)
        configs.append(('DistilBERT-3L-64d', cfg,
                        lambda c=cfg: DistilClf(c), (16, 64)))
    except Exception as e:
        print(f'[skip DistilBERT: {e}]')

    # T5 encoder
    try:
        from transformers import T5Config, T5EncoderModel
        class T5Clf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = T5EncoderModel(cfg)
                self.head = nn.Linear(cfg.d_model, 10)
            def forward(self, x):
                return self.head(self.model(inputs_embeds=x).last_hidden_state.mean(1))

        cfg = T5Config(d_model=64, num_heads=4, num_layers=2, d_ff=128,
                       vocab_size=100, d_kv=16)
        configs.append(('T5Enc-2L-64d', cfg,
                        lambda c=cfg: T5Clf(c), (16, 64)))
    except Exception as e:
        print(f'[skip T5: {e}]')

    # Electra
    try:
        from transformers import ElectraConfig, ElectraModel
        class ElectraClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = ElectraModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                return self.head(self.model(inputs_embeds=x).last_hidden_state[:, 0])

        cfg = ElectraConfig(hidden_size=64, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=128,
                            vocab_size=100, max_position_embeddings=32,
                            embedding_size=64)
        configs.append(('Electra-2L-64d', cfg,
                        lambda c=cfg: ElectraClf(c), (16, 64)))
    except Exception as e:
        print(f'[skip Electra: {e}]')

    # RoBERTa
    try:
        from transformers import RobertaConfig, RobertaModel
        class RoBERTaClf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = RobertaModel(cfg)
                self.head = nn.Linear(cfg.hidden_size, 10)
            def forward(self, x):
                return self.head(self.model(inputs_embeds=x).last_hidden_state[:, 0])

        cfg = RobertaConfig(hidden_size=64, num_hidden_layers=4,
                            num_attention_heads=4, intermediate_size=128,
                            vocab_size=100, max_position_embeddings=34,
                            type_vocab_size=1)
        configs.append(('RoBERTa-4L-64d', cfg,
                        lambda c=cfg: RoBERTaClf(c), (16, 64)))
    except Exception as e:
        print(f'[skip RoBERTa: {e}]')

    return configs


def run_benchmark():
    from developer_api import init_hf, check

    configs = build_hf_configs()

    # Add custom configs where HF default is likely suboptimal
    try:
        from transformers import GPT2Config, GPT2Model
        class GPT2Clf(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.gpt2 = GPT2Model(cfg)
                self.head = nn.Linear(cfg.n_embd, 10)
            def forward(self, x):
                return self.head(self.gpt2(inputs_embeds=x).last_hidden_state[:, -1])

        # Bad init: very high initializer_range
        cfg_hi = GPT2Config(n_embd=64, n_layer=4, n_head=4, vocab_size=100,
                            n_positions=32, initializer_range=0.1)
        configs.append(('GPT2-4L-highInit', cfg_hi,
                        lambda c=cfg_hi: GPT2Clf(c), (16, 64)))

        # Bad init: very low initializer_range
        cfg_lo = GPT2Config(n_embd=64, n_layer=4, n_head=4, vocab_size=100,
                            n_positions=32, initializer_range=0.002)
        configs.append(('GPT2-4L-lowInit', cfg_lo,
                        lambda c=cfg_lo: GPT2Clf(c), (16, 64)))
    except Exception:
        pass

    print(f'Benchmarking init_hf() on {len(configs)} HuggingFace configs\n')

    results = []
    n_steps = 80  # fast benchmark

    for i, (name, cfg, builder, input_shape) in enumerate(configs):
        print(f'[{i+1}/{len(configs)}] {name}', end=' ', flush=True)
        X, y = make_data(input_shape, n_samples=32, seed=42)

        # ── A: HF default init ──
        t0 = time.time()
        model_default = builder()
        default_losses = train(model_default, X, y, n_steps=n_steps, seed=42)
        t_default = time.time() - t0
        default_final = default_losses[-1] if default_losses else float('inf')

        # ── B: PhaseKit init plan ──
        plan = init_hf(cfg, verbose=False)

        # ── C: Apply PhaseKit init and train ──
        t0 = time.time()
        model_pk = builder()
        init_hf(cfg, model=model_pk, apply=True, verbose=False)
        pk_losses = train(model_pk, X, y, n_steps=n_steps, seed=42)
        t_pk = time.time() - t0
        pk_final = pk_losses[-1] if pk_losses else float('inf')

        # ── Metrics ──
        s_default = steps_to_threshold(default_losses, 1.0)
        s_pk = steps_to_threshold(pk_losses, 1.0)
        speedup = s_default / max(s_pk, 1) if s_pk < len(pk_losses) else 0

        winner = 'phasekit' if pk_final < default_final - 1e-4 else ('tie' if abs(pk_final - default_final) < 1e-3 else 'default')

        r = {
            'name': name,
            'model_type': plan.model_type,
            'depth': plan.depth,
            'width': plan.width,
            'activation': plan.activation,
            'norm_type': plan.norm_type,
            'n_params': plan.n_params,
            'default_sigma': plan.sigma_w_default,
            'pk_sigma_attn': plan.sigma_w_attn,
            'pk_sigma_ffn': plan.sigma_w_ffn,
            'pk_sigma_embed': plan.sigma_w_embed,
            'pk_sigma_head': plan.sigma_w_head,
            'default_final_loss': round(default_final, 6) if np.isfinite(default_final) else 'inf',
            'pk_final_loss': round(pk_final, 6) if np.isfinite(pk_final) else 'inf',
            'default_steps_to_1': s_default,
            'pk_steps_to_1': s_pk,
            'convergence_speedup': round(speedup, 2),
            'winner': winner,
            'default_time_s': round(t_default, 1),
            'pk_time_s': round(t_pk, 1),
        }
        results.append(r)

        d_str = f'{default_final:.4f}' if np.isfinite(default_final) else 'inf'
        p_str = f'{pk_final:.4f}' if np.isfinite(pk_final) else 'inf'
        w = '✓PK' if winner == 'phasekit' else ('≈' if winner == 'tie' else 'HF')
        sp = f' ({speedup:.1f}× faster)' if speedup > 1.05 else ''
        print(f'  {w}  default={d_str}  phasekit={p_str}  '
              f'steps-to-1: {s_default}→{s_pk}{sp}')

    # ── Summary ──
    pk_wins = sum(1 for r in results if r['winner'] == 'phasekit')
    ties = sum(1 for r in results if r['winner'] == 'tie')
    hf_wins = sum(1 for r in results if r['winner'] == 'default')
    speedups = [r['convergence_speedup'] for r in results if r['convergence_speedup'] > 0]
    mean_speedup = np.mean(speedups) if speedups else 0

    print(f'\n{"="*60}')
    print(f'HF INIT BENCHMARK RESULTS ({len(results)} configs)')
    print(f'{"="*60}')
    print(f'PhaseKit wins: {pk_wins}/{len(results)}')
    print(f'Ties:          {ties}/{len(results)}')
    print(f'HF wins:       {hf_wins}/{len(results)}')
    if speedups:
        print(f'Mean convergence speedup: {mean_speedup:.2f}×')
        print(f'Max convergence speedup:  {max(speedups):.2f}×')
    print(f'{"="*60}')

    summary = {
        'n_configs': len(results),
        'pk_wins': pk_wins,
        'ties': ties,
        'hf_wins': hf_wins,
        'mean_convergence_speedup': round(mean_speedup, 2),
        'max_convergence_speedup': round(max(speedups), 2) if speedups else 0,
        'details': results,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'hf_init_benchmark')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'hf_init_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')
    return summary


if __name__ == '__main__':
    run_benchmark()
