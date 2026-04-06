from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

DEFAULT_CORPUS = [
    "Transformers predict the next token from the context that came before it.",
    "Attention lets each token read from other tokens in the sequence.",
    "Feed-forward networks reshape each token independently after attention.",
    "Layer normalization keeps activations in a trainable numerical range.",
    "Tokenization turns raw text into discrete symbols before the model sees it.",
    "Chat fine-tuning changes the training distribution, not the architecture.",
    "A KV cache speeds up decoding by reusing previously computed keys and values.",
    "Temperature and top-k change how we sample from the logits at inference time.",
]

DEFAULT_CHAT_MESSAGES = [
    {"role": "system", "content": "You are a compact teaching assistant for an LLM course."},
    {"role": "user", "content": "Explain self-attention in three short sentences."},
]


def default_artifact_dir() -> Path:
    return Path("/Users/montekkundan/Developer/ML/llm/artifacts")


@dataclass
class TinyConfig:
    vocab_size: int
    block_size: int = 64
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], special_tokens: list[str] | None = None):
        self.stoi = stoi
        self.itos = {idx: token for token, idx in stoi.items()}
        self.special_tokens = special_tokens or SPECIAL_TOKENS

    @classmethod
    def build(cls, texts: Iterable[str], special_tokens: list[str] | None = None) -> "CharTokenizer":
        special_tokens = special_tokens or SPECIAL_TOKENS
        vocab = []
        for token in special_tokens:
            if token not in vocab:
                vocab.append(token)
        chars = sorted(set("".join(texts)))
        for ch in chars:
            if ch not in vocab:
                vocab.append(ch)
        stoi = {token: idx for idx, token in enumerate(vocab)}
        return cls(stoi=stoi, special_tokens=special_tokens)

    @property
    def pad_id(self) -> int:
        return self.stoi["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<|eos|>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, object]:
        return {"stoi": self.stoi, "special_tokens": self.special_tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharTokenizer":
        return cls(stoi={k: int(v) for k, v in payload["stoi"].items()}, special_tokens=list(payload["special_tokens"]))


def format_messages(messages: list[dict[str, str]], add_assistant_prompt: bool = False) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"<|{message['role']}|>\n{message['content'].strip()}\n")
    if add_assistant_prompt:
        lines.append("<|assistant|>\n")
    return "".join(lines)


class TinySelfAttention(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            is_causal = False
        else:
            is_causal = True

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.proj(attn)
        new_kv = (k.detach(), v.detach()) if use_cache else None
        return out, new_kv


class TinyMLP(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class TinyBlock(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = TinySelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = TinyMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyTransformerLM(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(TinyBlock(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        batch, seq_len = idx.shape
        past_len = 0
        if past_kvs:
            past_len = past_kvs[0][0].size(-2)
        positions = torch.arange(past_len, past_len + seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]

        new_past_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            block_past = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, past_kv=block_past, use_cache=use_cache)
            if use_cache and new_kv is not None:
                new_past_kvs.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss, new_past_kvs if use_cache else None


def build_demo_bundle(
    seed: int = 0,
    steps: int = 80,
    learning_rate: float = 3e-3,
    block_size: int = 64,
    device: str = "cpu",
    corpus: list[str] | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    corpus = corpus or DEFAULT_CORPUS
    train_text = "\n".join(corpus * 16)
    eval_text = "\n".join(corpus[:4] * 4)
    tokenizer = CharTokenizer.build([train_text, format_messages(DEFAULT_CHAT_MESSAGES, add_assistant_prompt=True)])
    config = TinyConfig(vocab_size=len(tokenizer.stoi), block_size=block_size)
    model = TinyTransformerLM(config).to(device)
    history = train_model(
        model=model,
        tokenizer=tokenizer,
        train_text=train_text,
        eval_text=eval_text,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=8,
        device=device,
    )
    metadata = {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "train_corpus_size": len(train_text),
        "eval_corpus_size": len(eval_text),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"model": model, "tokenizer": tokenizer, "config": config, "history": history, "metadata": metadata}


def _batch_from_text(tokenizer: CharTokenizer, text: str, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=torch.long)
    if len(ids) < block_size + 1:
        raise ValueError("text is too short for the chosen block_size")
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([ids[s : s + block_size] for s in starts]).to(device)
    y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts]).to(device)
    return x, y


def train_model(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    train_text: str,
    eval_text: str,
    steps: int = 80,
    learning_rate: float = 3e-3,
    batch_size: int = 8,
    device: str = "cpu",
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        x, y = _batch_from_text(tokenizer, train_text, model.config.block_size, batch_size=batch_size, device=device)
        _, loss, _ = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            metrics = evaluate_model(model, tokenizer, eval_text, batch_size=batch_size, device=device)
            history.append(
                {
                    "step": float(step),
                    "train_loss": float(loss.item()),
                    "eval_loss": float(metrics["loss"]),
                    "bpb": float(metrics["bpb"]),
                }
            )
    return history


@torch.no_grad()
def evaluate_model(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    text: str,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    x, y = _batch_from_text(tokenizer, text, model.config.block_size, batch_size=batch_size, device=device)
    _, loss, _ = model(x, targets=y)
    bpb = float(loss.item() / math.log(2))
    model.train()
    return {"loss": float(loss.item()), "bpb": bpb}


def save_checkpoint(
    path: str | Path,
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    config: TinyConfig,
    metadata: dict[str, object] | None = None,
    history: list[dict[str, float]] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "tokenizer": tokenizer.to_dict(),
            "metadata": metadata or {},
            "history": history or [],
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, object]:
    payload = torch.load(Path(path), map_location=device)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    config = TinyConfig(**payload["config"])
    model = TinyTransformerLM(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "metadata": payload.get("metadata", {}),
        "history": payload.get("history", []),
        "path": str(path),
    }


def ensure_demo_checkpoint(
    path: str | Path | None = None,
    steps: int = 80,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, object]:
    path = Path(path or default_artifact_dir() / "demo" / "tiny_transformer.pt")
    if path.exists():
        return load_checkpoint(path, device=device)
    bundle = build_demo_bundle(seed=seed, steps=steps, device=device)
    save_checkpoint(
        path=path,
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        config=bundle["config"],
        metadata=bundle["metadata"],
        history=bundle["history"],
    )
    return load_checkpoint(path, device=device)


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> int:
    logits = logits.clone()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k is not None and top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        cutoff = values[-1]
        logits[logits < cutoff] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def prefill_prompt(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str = "cpu",
) -> tuple[list[int], list[tuple[torch.Tensor, torch.Tensor]]]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if len(prompt_ids) > model.config.block_size:
        prompt_ids = prompt_ids[-model.config.block_size :]
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    _, _, past_kvs = model(idx, use_cache=True)
    return prompt_ids, past_kvs or []


@torch.no_grad()
def decode_next_token(
    model: TinyTransformerLM,
    current_token_id: int,
    past_kvs: list[tuple[torch.Tensor, torch.Tensor]],
    temperature: float = 0.8,
    top_k: int | None = 8,
    device: str = "cpu",
) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    max_cache = model.config.block_size - 1
    cropped_past_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for past_k, past_v in past_kvs:
        if past_k.size(-2) > max_cache:
            past_k = past_k[..., -max_cache:, :]
            past_v = past_v[..., -max_cache:, :]
        cropped_past_kvs.append((past_k, past_v))
    idx = torch.tensor([[current_token_id]], dtype=torch.long, device=device)
    logits, _, past_kvs = model(idx, past_kvs=cropped_past_kvs, use_cache=True)
    next_id = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
    return next_id, past_kvs or [], logits[0, -1]


@torch.no_grad()
def generate_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = 8,
    device: str = "cpu",
) -> str:
    prompt_ids, past_kvs = prefill_prompt(model, tokenizer, prompt, device=device)
    current_token_id = prompt_ids[-1]
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        next_id, past_kvs, _ = decode_next_token(
            model,
            current_token_id=current_token_id,
            past_kvs=past_kvs,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        if next_id == tokenizer.eos_id:
            break
        new_ids.append(next_id)
        current_token_id = next_id
    return tokenizer.decode(new_ids)


@torch.no_grad()
def stream_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = 8,
    device: str = "cpu",
):
    prompt_ids, past_kvs = prefill_prompt(model, tokenizer, prompt, device=device)
    current_token_id = prompt_ids[-1]
    for _ in range(max_new_tokens):
        next_id, past_kvs, _ = decode_next_token(
            model,
            current_token_id=current_token_id,
            past_kvs=past_kvs,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        if next_id == tokenizer.eos_id:
            break
        piece = tokenizer.decode([next_id])
        yield piece
        current_token_id = next_id


def available_checkpoints(directory: str | Path | None = None) -> list[dict[str, object]]:
    directory = Path(directory or default_artifact_dir())
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu")
            items.append(
                {
                    "path": str(path),
                    "config": payload.get("config", {}),
                    "metadata": payload.get("metadata", {}),
                }
            )
        except Exception:
            items.append({"path": str(path), "config": {}, "metadata": {"error": "unreadable"}})
    return items


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
