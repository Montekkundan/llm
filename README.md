# LLM From Scratch and Deploy

This repo is the runnable companion to the Obsidian lecture notes under `lectures/LLM`.

Teaching flow:

- the Obsidian notes carry the full theory and lecture narrative
- this repo carries the runnable notebooks and demo scripts
- Rasbt and nanochat are used at the end of each lecture as reference implementations after the core concept is already clear

## Lecture Companion Map

Use the lecture note for theory and the mapped repo artifact below for live code demos.

### Concept Notes

| Lecture note | Primary artifact | Repo path | Status |
| --- | --- | --- | --- |
| Tokenisation | Notebook + scripts | `notebooks/tokenization/lecture_walkthrough.ipynb`, `scripts/tokenization/` | Ready |
| Embedding Layer | Notebook + scripts | `notebooks/embedding_layer/lecture_walkthrough.ipynb`, `scripts/embedding_layer/` | Ready |
| Positional Encoding | Notebook + scripts | `notebooks/positional_encoding/lecture_walkthrough.ipynb`, `scripts/positional_encoding/` | Ready |
| Scaled Dot-Product Attention | Notebook + scripts | `notebooks/scaled_dot_product_attention/lecture_walkthrough.ipynb`, `scripts/scaled_dot_product_attention/` | Ready |
| Multi-head Attention | Notebook + scripts | `notebooks/multi_head_attention/lecture_walkthrough.ipynb`, `scripts/multi_head_attention/` | Ready |
| Feed-Forward Network | Notebook + scripts | `notebooks/feed_forward_network/lecture_walkthrough.ipynb`, `scripts/feed_forward_network/` | Ready |
| Layer Normalization | Notebook + scripts | `notebooks/layer_normalization/lecture_walkthrough.ipynb`, `scripts/layer_normalization/` | Ready |
| Encoder Block | Notebook + scripts | `notebooks/encoder_block/lecture_walkthrough.ipynb`, `scripts/encoder_block/` | Ready |
| Decoder Block | Notebook + scripts | `notebooks/decoder_block/lecture_walkthrough.ipynb`, `scripts/decoder_block/` | Ready |
| Causal Language Modeling | Notebook + scripts | `notebooks/causal_language_modeling/lecture_walkthrough.ipynb`, `scripts/causal_language_modeling/` | Ready |
| Chat Format and SFT | Notebook + scripts | `notebooks/chat_format_and_sft/lecture_walkthrough.ipynb`, `scripts/chat_format_and_sft/` | Ready |
| Inference and Sampling | Notebook + scripts | `notebooks/inference_and_sampling/lecture_walkthrough.ipynb`, `scripts/inference_and_sampling/` | Ready |
| Training Loop | Notebook + scripts | `notebooks/training_loop/lecture_walkthrough.ipynb`, `scripts/training_loop/` | Ready |
| LoRA and Parameter-Efficient Fine-Tuning | Notebook + scripts | `notebooks/lora_and_parameter_efficient_fine_tuning/lecture_walkthrough.ipynb`, `scripts/lora_and_parameter_efficient_fine_tuning/` | Ready |
| Quantization | Notebook + scripts | `notebooks/quantization/lecture_walkthrough.ipynb`, `scripts/quantization/` | Ready |
| FastAPI Chat App | Python demo + UI | `scripts/fastapi_chat_app/` | Ready |
| Deployment | Python/demo docs | `scripts/deployment/`, `RUN_APP.md`, `Dockerfile` | Ready |

### Product Notes

| Lecture note | Primary artifact | Repo path | Status |
| --- | --- | --- | --- |
| Nanochat Architecture | Notebook + scripts | `notebooks/nanochat_architecture/lecture_walkthrough.ipynb`, `scripts/nanochat_architecture/` | Ready |
| Base Training Flow | Python demos | `scripts/base_training_flow/` | Ready |
| Base Evaluation Flow | Python demos | `scripts/base_evaluation_flow/` | Ready |
| CLI and Web Chat | Python demos + UI | `scripts/cli_and_web_chat/` | Ready |
| Inference Runtime and KV Cache | Notebook + scripts | `notebooks/inference_runtime_and_kv_cache/lecture_walkthrough.ipynb`, `scripts/inference_runtime_and_kv_cache/` | Ready |
| SFT Flow | Notebook + scripts | `notebooks/sft_flow/lecture_walkthrough.ipynb`, `scripts/sft_flow/` | Ready |
| Real Chatbot Workflow | Notebook + scripts | `notebooks/real_chatbot_workflow/lecture_walkthrough.ipynb`, `scripts/real_chatbot_workflow/`, `picollm/` | Ready |

## Reference Repos

- Rasbt: concept-first step-by-step implementations and notebooks
- nanochat: product-oriented training, evaluation, inference, and chat system code

## Serious Demo Track

For the final lecture demo, use [picollm/](/Users/montekkundan/Developer/ML/llm/picollm/README.md).
The single best student run doc is [picollm/RUNBOOK.md](/Users/montekkundan/Developer/ML/llm/picollm/RUNBOOK.md).

That track is intentionally separate from the tiny from-scratch teaching model:

- `picollm/pretrain_cloud/`: train your own checkpoint on rented GPUs
- `picollm/sft_lora/`: run a serious LoRA fine-tuning demo
- `picollm/serve/`: run the tuned or pretrained chatbot locally on `cuda`, `mps`, or `cpu`
- `picollm/pretrain_cloud/VAST_AI_SETUP.md`: CLI-first Vast.ai setup
- `picollm/HUGGING_FACE_SETUP.md`: token/login/push/pull workflow

## Acknowledgements

- This project is strongly inspired by Andrej Karpathy's [`nanochat`](https://github.com/karpathy/nanochat).
- Thank you to [Hugging Face](https://huggingface.co/) for the open tooling and datasets ecosystem that make projects like this easier to teach and build.

## Cite

If you want to cite the inspiration behind the application workflow, cite Karpathy's `nanochat`:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

If you want to cite this repo itself, use:

```bibtex
@misc{llm_from_scratch_and_deploy,
  author = {Montek Kundan},
  title = {LLM From Scratch and Deploy},
  year = {2026},
  publisher = {GitHub}
}
```

## License

MIT. See [LICENSE](/Users/montekkundan/Developer/ML/llm/LICENSE).
