from __future__ import annotations

import torch
import torch.nn as nn


def one_hot_equals_lookup() -> None:
    vocab = ["<pad>", "I", "like", "LLMs", "."]
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    E = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.5],
            [0.2, 1.2, 0.3],
            [1.5, -0.2, 0.7],
            [-0.3, 0.4, 1.1],
        ]
    )
    token_id = token_to_id["LLMs"]
    one_hot = torch.nn.functional.one_hot(torch.tensor(token_id), num_classes=len(vocab)).float()
    print("one_hot @ E:", one_hot @ E)
    print("row lookup :", E[token_id])


def padding_gradient_demo() -> None:
    vocab = ["<pad>", "I", "like", "LLMs", "."]
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    batch_ids = torch.tensor(
        [
            [token_to_id["I"], token_to_id["like"], token_to_id["LLMs"], token_to_id["."]],
            [token_to_id["I"], token_to_id["LLMs"], token_to_id["<pad>"], token_to_id["<pad>"]],
        ]
    )
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4, padding_idx=token_to_id["<pad>"])
    embedding(batch_ids).sum().backward()
    print("grad for <pad> row:")
    print(embedding.weight.grad[token_to_id["<pad>"]])
    print("grad for 'I' row:")
    print(embedding.weight.grad[token_to_id["I"]])


def weight_tying_demo() -> None:
    class TinyTiedLM(nn.Module):
        def __init__(self, vocab_size: int, d_model: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            hidden = self.embed(ids).mean(dim=1)
            return self.lm_head(hidden)

    model = TinyTiedLM(vocab_size=10, d_model=6)
    ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    hidden = model.embed(ids).mean(dim=1)
    print("parameter count:", sum(p.numel() for p in model.parameters()))
    print("head == hidden @ embedding.T:", torch.allclose(model(ids), hidden @ model.embed.weight.T))


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== one-hot equals lookup ==")
    one_hot_equals_lookup()
    print("\n== padding row gets no gradient ==")
    padding_gradient_demo()
    print("\n== weight tying ==")
    weight_tying_demo()
