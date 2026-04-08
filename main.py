CONCEPTS = [
    "tokenization",
    "embedding_layer",
    "positional_encoding",
    "scaled_dot_product_attention",
    "multi_head_attention",
    "feed_forward_network",
    "layer_normalization",
    "encoder_block",
    "decoder_block",
    "causal_language_modeling",
    "chat_format_and_sft",
    "inference_and_sampling",
    "training_loop",
    "lora_and_parameter_efficient_fine_tuning",
    "quantization",
]

PRODUCT = [
    "nanochat_architecture",
    "base_training_flow",
    "base_evaluation_flow",
    "cli_and_web_chat",
    "inference_runtime_and_kv_cache",
    "sft_flow",
    "real_chatbot_workflow",
    "fastapi_chat_app",
    "deployment",
]


def main():
    print("LLM Concepts Course Companion")
    print("\nTheory layer:")
    for slug in CONCEPTS:
        print(f"\n[{slug}]")
        print(f"- notebook: notebooks/{slug}/lecture_walkthrough.ipynb")
    print("\nProduct layer:")
    for slug in PRODUCT:
        print(f"- {slug}")
    print("\nPrimary teaching path: open notebooks with: uv run jupyter lab")


if __name__ == "__main__":
    main()
