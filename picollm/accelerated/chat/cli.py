import argparse
from pathlib import Path
import torch
from picollm.accelerated.common import compute_init, autodetect_device_type
from picollm.accelerated.engine import Engine
from picollm.accelerated.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", choices=['base', 'sft'], help="Source of the model: base|sft")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--top-p', type=float, default=None, help='Top-p sampling parameter. empty => disabled')
parser.add_argument('--min-p', type=float, default=None, help='Min-p cutoff relative to the most likely token. empty => disabled')
parser.add_argument('--max-tokens', type=int, default=256, help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42, help='Random seed used for sampling')
system_group = parser.add_mutually_exclusive_group()
system_group.add_argument('--system-prompt', type=str, default='', help='System instruction text to prepend to the conversation')
system_group.add_argument('--system-file', type=str, default='', help='Path to a file containing the system instruction text')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

engine = Engine(model, tokenizer)


def read_system_prompt() -> str:
    if args.system_prompt:
        return args.system_prompt.strip()
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8").strip()
    return ""


def append_user_turn(tokens: list[int], text: str):
    tokens.append(user_start)
    tokens.extend(tokenizer.encode(text))
    tokens.append(user_end)


def generation_budget_error(prompt_tokens: list[int]) -> str | None:
    context_window = engine.model.config.sequence_len
    prompt_length = len(prompt_tokens)
    remaining_tokens = context_window - prompt_length
    if remaining_tokens <= 0:
        return (
            f"Conversation is already {prompt_length} tokens after formatting, which exceeds the "
            f"{context_window}-token context window. Start a new conversation or shorten the history."
        )
    if args.max_tokens > remaining_tokens:
        return (
            f"Requested --max-tokens={args.max_tokens}, but only {remaining_tokens} tokens remain in the "
            f"{context_window}-token context window. Start a new conversation or lower --max-tokens."
        )
    return None


print("\npicoLLM Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]
system_prompt = read_system_prompt()
if system_prompt:
    append_user_turn(conversation_tokens, f"System instruction:\n{system_prompt}")

while True:

    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        if system_prompt:
            append_user_turn(conversation_tokens, f"System instruction:\n{system_prompt}")
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    candidate_tokens = conversation_tokens.copy()
    append_user_turn(candidate_tokens, user_input)
    candidate_tokens.append(assistant_start)
    budget_error = generation_budget_error(candidate_tokens)
    if budget_error is not None:
        print(f"\nContext budget error: {budget_error}")
        if args.prompt:
            raise SystemExit(1)
        continue

    conversation_tokens = candidate_tokens
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
