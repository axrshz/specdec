"""Minimal GPU benchmark: speculative_decode vs vanilla autoregressive."""

import argparse
import time
import torch
from transformers import AutoTokenizer

from main import draft_name, load_model, speculative_decode, target_name


@torch.inference_mode()
def baseline_generate(model, input_ids, max_new_tokens=100, temperature=1.0):
    """Vanilla greedy/sampling autoregressive decode with KV-cache. Batch 1 only."""
    assert input_ids.shape[0] == 1
    eos_ids = torch.tensor(
        model.generation_config.eos_token_id, dtype=torch.long, device=input_ids.device
    )

    def _probs(logits):
        if temperature == 0.0:
            p = torch.zeros_like(logits)
            p.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            return p
        return torch.softmax(logits / temperature, dim=-1)

    def _sample(logits):
        p = _probs(logits)
        if temperature == 0.0:
            return p.argmax(dim=-1, keepdim=True)
        return torch.multinomial(p, num_samples=1)

    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    sequence = input_ids.clone()
    generated = 0

    while generated < max_new_tokens:
        nxt = _sample(logits)
        # EOS check
        if torch.isin(nxt[0], eos_ids).any():
            sequence = torch.cat([sequence, nxt], dim=1)
            generated += 1
            break
        sequence = torch.cat([sequence, nxt], dim=1)
        generated += 1
        if generated >= max_new_tokens:
            break
        out = model(input_ids=nxt, past_key_values=cache)
        logits = out.logits[:, -1, :]
        cache = out.past_key_values

    return sequence[:, input_ids.shape[1] :]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_run(fn, *args, **kwargs):
    _sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    t1 = time.perf_counter()
    return out, t1 - t0


def benchmark(
    prompts,
    max_new_tokens=100,
    draft_tokens=5,
    temperature=0.0,
    num_runs=3,
    warmup=1,
):
    print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)} | dtype=bfloat16")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: no GPU detected - timings will be CPU-bound and not representative")

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    print(f"loading target {target_name} ...")
    target_model = load_model(target_name)
    print(f"loading draft {draft_name} ...")
    draft_model = load_model(draft_name)
    device = target_model.device
    print(f"target device: {device} | draft device: {draft_model.device}")

    results = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]
        print(f"\n=== prompt ({prompt_len} tokens): {prompt[:80]!r} ===")
        print(f"max_new_tokens={max_new_tokens} draft_tokens={draft_tokens} T={temperature}")

        # warmup (compile caches, etc)
        for _ in range(warmup):
            baseline_generate(target_model, input_ids, max_new_tokens, temperature)
            speculative_decode(target_model, draft_model, input_ids, max_new_tokens, draft_tokens, temperature)
            _sync()

        # baseline
        b_times = []
        for _ in range(num_runs):
            _, dt = timed_run(baseline_generate, target_model, input_ids, max_new_tokens, temperature)
            b_times.append(dt)
        b_avg = sum(b_times) / len(b_times)
        b_tps = max_new_tokens / b_avg if b_avg > 0 else 0

        # speculative
        s_times = []
        for _ in range(num_runs):
            _, dt = timed_run(speculative_decode, target_model, draft_model, input_ids, max_new_tokens, draft_tokens, temperature)
            s_times.append(dt)
        s_avg = sum(s_times) / len(s_times)
        s_tps = max_new_tokens / s_avg if s_avg > 0 else 0

        speedup = b_avg / s_avg if s_avg > 0 else float("inf")

        # one correctness sample (decode)
        b_out = baseline_generate(target_model, input_ids, max_new_tokens, temperature)
        s_out = speculative_decode(target_model, draft_model, input_ids, max_new_tokens, draft_tokens, temperature)
        print(f"baseline : {b_avg:.3f}s avg over {num_runs} runs | {b_tps:.1f} tok/s | {b_out.shape[1]} tokens")
        print(f"speculative: {s_avg:.3f}s avg over {num_runs} runs | {s_tps:.1f} tok/s | {s_out.shape[1]} tokens")
        print(f"speedup: {speedup:.2f}x")
        # show raw times for variance
        print(f"  baseline times: {[f'{x:.3f}' for x in b_times]}")
        print(f"  spec times    : {[f'{x:.3f}' for x in s_times]}")
        # truncated decode preview
        print(f"  baseline preview: {tokenizer.decode(b_out[0][:40], skip_special_tokens=True)[:120]!r}")
        print(f"  spec preview    : {tokenizer.decode(s_out[0][:40], skip_special_tokens=True)[:120]!r}")

        if torch.cuda.is_available():
            print(f"  peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        results.append((prompt, b_avg, s_avg, speedup))

    print("\n=== summary ===")
    for prompt, b_avg, s_avg, speedup in results:
        print(f"{speedup:.2f}x | baseline {b_avg:.3f}s | spec {s_avg:.3f}s | {prompt[:60]!r}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--draft-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0=greedy")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Explain speculative decoding in simple terms.")
    parser.add_argument("--prompts-file", type=str, default=None, help="one prompt per line")
    args = parser.parse_args()

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = [args.prompt]

    benchmark(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        draft_tokens=args.draft_tokens,
        temperature=args.temperature,
        num_runs=args.num_runs,
        warmup=args.warmup,
    )
