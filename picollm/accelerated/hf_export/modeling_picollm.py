from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_picollm import PicoLlmConfig
except ImportError:
    from configuration_picollm import PicoLlmConfig


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_value_embedding(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_dim_half = x.shape[-1] // 2
    x1, x2 = x[..., :head_dim_half], x[..., head_dim_half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class PicoLlmAttention(nn.Module):
    def __init__(self, config: PicoLlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = CastLinear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = CastLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = CastLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = CastLinear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = (
            CastLinear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_value_embedding(layer_idx, config.n_layer)
            else None
        )

    def _attention_bias(
        self,
        q_len: int,
        attention_mask: torch.Tensor | None,
        window_size: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(q_len, device=device)
        allowed = positions.unsqueeze(1) >= positions.unsqueeze(0)
        left_window, _ = window_size
        if left_window >= 0:
            allowed = allowed & ((positions.unsqueeze(1) - positions.unsqueeze(0)) < left_window)
        allowed = allowed.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            key_allowed = attention_mask.to(torch.bool)[:, None, None, :]
            allowed = allowed & key_allowed
        bias = torch.zeros(allowed.shape, dtype=dtype, device=device)
        bias.masked_fill_(~allowed, torch.finfo(dtype).min)
        return bias

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q) * 1.2
        k = rms_norm(k) * 1.2

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.n_kv_head != self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_bias = self._attention_bias(
            q_len=seq_len,
            attention_mask=attention_mask,
            window_size=window_size,
            dtype=q.dtype,
            device=q.device,
        )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return self.c_proj(y)


class PicoLlmMLP(nn.Module):
    def __init__(self, config: PicoLlmConfig) -> None:
        super().__init__()
        self.c_fc = CastLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = CastLinear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class PicoLlmBlock(nn.Module):
    def __init__(self, config: PicoLlmConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = PicoLlmAttention(config, layer_idx)
        self.mlp = PicoLlmMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), ve, cos_sin, window_size, attention_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class PicoLlmForCausalLM(PreTrainedModel):
    config_class = PicoLlmConfig
    base_model_prefix = "transformer"
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config: PicoLlmConfig) -> None:
        super().__init__(config)
        padded_vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [PicoLlmBlock(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = CastLinear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = CastLinear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_value_embedding(i, config.n_layer)
            }
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        n_embd = self.config.n_embd
        init_scale = (3**0.5) * n_embd**-0.5
        if isinstance(module, nn.Embedding):
            if module.weight.shape[1] == self.config.n_embd:
                nn.init.normal_(module.weight, mean=0.0, std=0.8)
            else:
                nn.init.uniform_(module.weight, -init_scale, init_scale)
            return
        if not isinstance(module, CastLinear):
            return
        out_features, in_features = module.weight.shape
        if out_features == self.config.vocab_size or in_features == self.config.n_embd and out_features == 1:
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            return
        if out_features == self.config.n_embd and in_features != self.config.n_embd:
            nn.init.zeros_(module.weight)
            return
        if out_features == 4 * self.config.n_embd:
            nn.init.uniform_(module.weight, -init_scale * 0.4, init_scale * 0.4)
            return
        nn.init.uniform_(module.weight, -init_scale, init_scale)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 100000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.transformer["wte"].weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos, sin

    def refresh_rotary_embeddings(self) -> None:
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin

    def _compute_window_sizes(self, config: PicoLlmConfig) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()
        if any(char not in "SL" for char in pattern):
            raise ValueError(f"Invalid window_pattern: {pattern}")
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer["wte"]

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer["wte"] = value

    def get_output_embeddings(self) -> CastLinear:
        return self.lm_head

    def set_output_embeddings(self, value: CastLinear) -> None:
        self.lm_head = value

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_len = input_ids.shape
        if seq_len > self.cos.size(1):
            raise ValueError(
                f"Sequence length grew beyond the rotary cache: {seq_len} > {self.cos.size(1)}"
            )
        cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]
        x = self.transformer["wte"](input_ids)
        x = rms_norm(x)
        if seq_len > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        x_backout = None
        backout_layer = self.config.n_layer // 2
        for layer_idx, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            ve = (
                self.value_embeds[str(layer_idx)](input_ids).to(x.dtype)
                if str(layer_idx) in self.value_embeds
                else None
            )
            x = block(x, ve, cos_sin, self.window_sizes[layer_idx], attention_mask)
            if layer_idx == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = rms_norm(x)

        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, None)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
