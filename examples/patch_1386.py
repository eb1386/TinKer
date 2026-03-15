"""patch 1386 model"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

# make importable
tinker_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(tinker_root))

from tinker import patch_model


def load_1386_model(checkpoint_path: str, config_path: str):
    """load 1386 model"""
    # resolve root
    project_root = Path(config_path).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    from src.train.utils import load_checkpoint

    # load config
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # extract fields
    model_config_dict = raw_config.get("model", raw_config)
    config = ModelConfig.from_dict(model_config_dict)

    # build model
    model = Transformer(config)
    load_checkpoint(checkpoint_path, model)

    return model, config


@torch.no_grad()
def greedy_generate(model, prompt_tokens: torch.Tensor, max_new_tokens: int = 50):
    """greedy token generation"""
    tokens = prompt_tokens.clone()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def benchmark_forward(model, tokens: torch.Tensor, warmup: int = 5,
                      runs: int = 20) -> float:
    """measure forward latency"""
    # warmup
    for _ in range(warmup):
        _ = model(tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(
        description="Patch a 1386.ai model with TinKer and compare performance."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to 1386.ai checkpoint (e.g. ../1386.ai/checkpoints/final.pt)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to 1386.ai config YAML (e.g. ../1386.ai/configs/pretrain_1.0.yaml)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Number of tokens to generate for comparison (default: 50)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=64,
        help="Prompt sequence length for benchmarking (default: 64)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # load model
    print(f"Loading 1386.ai model from {args.checkpoint}...")
    model, config = load_1386_model(args.checkpoint, args.config)
    model = model.to(device=device, dtype=dtype).eval()

    # baseline generation
    prompt = torch.randint(1, config.vocab_size, (1, args.seq_len), device=device)

    print("Running baseline (PyTorch) generation...")
    baseline_tokens = greedy_generate(model, prompt, args.max_new_tokens)

    # benchmark baseline
    baseline_latency = benchmark_forward(model, prompt)
    print(f"  Baseline forward latency: {baseline_latency:.2f} ms")

    # patch model
    print("\nPatching model with TinKer...")
    patch_model(
        model,
        rope_theta=getattr(config, "rope_theta", 10000.0),
        max_seq_len=getattr(config, "max_seq_len", 2048),
        verbose=True,
    )

    # tinker generation
    print("\nRunning TinKer generation...")
    tinker_tokens = greedy_generate(model, prompt, args.max_new_tokens)

    # benchmark tinker
    tinker_latency = benchmark_forward(model, prompt)
    print(f"  TinKer forward latency:   {tinker_latency:.2f} ms")

    # compare outputs
    match = torch.equal(baseline_tokens, tinker_tokens)
    print(f"\n--- Results ---")
    print(f"Output match (greedy, temperature=0): {'PASS' if match else 'FAIL'}")
    if not match:
        diff_positions = (baseline_tokens != tinker_tokens).nonzero(as_tuple=True)
        print(f"  First mismatch at position: {diff_positions[1][0].item()}")
        print(f"  Baseline token: {baseline_tokens[0, diff_positions[1][0]].item()}")
        print(f"  TinKer token:   {tinker_tokens[0, diff_positions[1][0]].item()}")

    print(f"\nBaseline latency: {baseline_latency:.2f} ms")
    print(f"TinKer latency:   {tinker_latency:.2f} ms")
    speedup = baseline_latency / tinker_latency if tinker_latency > 0 else float("inf")
    print(f"Speedup:          {speedup:.2f}x")


if __name__ == "__main__":
    main()
