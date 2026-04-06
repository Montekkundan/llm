# LLM From Scratch and Deploy

This repository is the runnable companion to the lecture notes at `lectures.montek.dev`.

The course has two layers:

- theory and teaching narrative in your lecture notes
- runnable notebooks, scripts, and demo apps in this repo

## Start Here

Students should use the lecture notes for the full explanation, then open the linked code from the relevant lesson.

## Student Setup

Install the repo first:

```bash
uv sync
```

Optional but recommended if you plan to use Hugging Face models or push checkpoints:

```bash
hf auth login
```

Then use this workflow:

1. Open the relevant lesson on `lectures.montek.dev`.
2. Read the theory there first.
3. Open the linked notebook or script from this repo.
4. For the serious chatbot path, follow `picollm/RUNBOOK.md`.

For the serious chatbot path, start with:

- [picollm/RUNBOOK.md](/Users/montekkundan/Developer/ML/llm/picollm/RUNBOOK.md)

That runbook covers:

- baseline Qwen
- LoRA fine-tuning
- base vs LoRA comparison
- serving the adapted model
- Vast.ai helper commands
- bringing a cloud-trained checkpoint back to a local machine

## Repo Guide

Use these directories by purpose:

- `notebooks/`: live lecture walkthroughs
- `scripts/`: runnable demo scripts and small apps
- `apps/`: production-style frontend apps
- `course_tools/`: tiny from-scratch runtime used by the teaching demos
- `picollm/`: serious model workflow for the final chatbot lectures

## Final Demo Docs

For the final lecture sequence:

- [picollm/RUNBOOK.md](/Users/montekkundan/Developer/ML/llm/picollm/RUNBOOK.md)
- [picollm/HUGGING_FACE_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/HUGGING_FACE_SETUP.md)
- [picollm/pretrain_cloud/VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)
- [prompts/real_chatbot_workflow/base_vs_lora_prompts.md](/Users/montekkundan/Developer/ML/llm/prompts/real_chatbot_workflow/base_vs_lora_prompts.md)
- [apps/vercel_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/README.md)
- [apps/opentui_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/opentui_ai_sdk_chat/README.md)

## Reference Repos

- Rasbt: concept-first step-by-step implementations and notebooks
- nanochat: product-oriented training, evaluation, inference, and chat system code

## Acknowledgements

- This project is strongly inspired by Andrej Karpathy's [`nanochat`](https://github.com/karpathy/nanochat).
- This project also draws heavily on Sebastian Raschka's [`LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch).
- Thank you to [Hugging Face](https://huggingface.co/) for the open tooling and datasets ecosystem that make projects like this easier to teach and build.

## Cite

If you want to cite the reference repos behind the teaching flow, cite both Raschka and Karpathy:

```yaml
cff-version: 1.2.0
message: "If you use this book or its accompanying code, please cite it as follows."
title: "Build A Large Language Model (From Scratch), Published by Manning, ISBN 978-1633437166"
abstract: "This book provides a comprehensive, step-by-step guide to implementing a ChatGPT-like large language model from scratch in PyTorch."
date-released: 2024-09-12
authors:
  - family-names: "Raschka"
    given-names: "Sebastian"
license: "Apache-2.0"
url: "https://www.manning.com/books/build-a-large-language-model-from-scratch"
repository-code: "https://github.com/rasbt/LLMs-from-scratch"
keywords:
  - large language models
  - natural language processing
  - artificial intelligence
  - PyTorch
  - machine learning
  - deep learning
```

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
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
  publisher = {GitHub},
  url = {https://github.com/Montekkundan/llm}
}
```

## License

MIT. See [LICENSE](/Users/montekkundan/Developer/ML/llm/LICENSE).
