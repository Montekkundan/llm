from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPTS = [
    "Why is the sky blue?",
    "Explain self-attention to a beginner in four sentences.",
    "Write a short poem about the sky.",
    "Give me a two-step study plan for learning transformers.",
]


if __name__ == "__main__":
    print("Real chatbot workflow")
    print()
    print("1. Start from a real small instruct model.")
    print("2. Optionally run LoRA on course-specific chat data.")
    print("3. Evaluate base vs adapter behavior.")
    print("4. Serve the model locally through CLI or FastAPI.")
    print("5. Point students to nanochat for deeper optimization work.")
    print()
    print("Demo prompts:")
    for prompt in PROMPTS:
        print("-", prompt)
    print()
    print("Code map:")
    print("-", ROOT / "picollm" / "README.md")
    print("-", ROOT / "picollm" / "HUGGING_FACE_SETUP.md")
    print("-", ROOT / "picollm" / "pretrain_cloud" / "VAST_AI_SETUP.md")
    print("-", ROOT / "picollm" / "sft_lora" / "README.md")
    print("-", ROOT / "picollm" / "serve" / "README.md")
    print("-", ROOT / "scripts" / "real_chatbot_workflow" / "compare_base_vs_lora.py")
