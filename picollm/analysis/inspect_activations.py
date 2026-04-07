from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from picollm.common import load_generation_bundle
from picollm.common.chat import build_prompt


def _tensor_mean_norm(tensor: torch.Tensor) -> float:
    values = tensor.float()
    return float(values.norm(dim=-1).mean().item())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect hidden states and attention summaries for one prompt.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle = load_generation_bundle(args.model, adapter_path=args.adapter, device=args.device)
    prompt_text = build_prompt(bundle.tokenizer, [{"role": "user", "content": args.prompt}], add_generation_prompt=True)
    encoded = bundle.tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(bundle.model.device) for key, value in encoded.items()}

    outputs = bundle.model(
        **encoded,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
    )

    hidden_state_norms = [
        {
            "layer": index,
            "mean_token_norm": _tensor_mean_norm(hidden_state[0]),
        }
        for index, hidden_state in enumerate(outputs.hidden_states or [])
    ]

    attention_summaries = []
    for layer_index, attention in enumerate(outputs.attentions or []):
        # attention: [batch, heads, seq, seq]
        head_payload = []
        for head_index in range(attention.shape[1]):
            weights = attention[0, head_index].float()
            last_token_weights = weights[-1]
            top_index = int(torch.argmax(last_token_weights).item())
            head_payload.append(
                {
                    "head": head_index,
                    "last_token_top_source_index": top_index,
                    "last_token_top_source_weight": float(last_token_weights[top_index].item()),
                }
            )
        attention_summaries.append({"layer": layer_index, "heads": head_payload})

    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "prompt": args.prompt,
        "prompt_text": prompt_text,
        "prompt_tokens": bundle.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
        "hidden_state_norms": hidden_state_norms,
        "attention_summary": attention_summaries,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

