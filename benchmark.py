"""Minimal GPU benchmark: speculative_decode vs vanilla autoregressive."""

import argparse
import time
import torch
from transformers import AutoTokenizer

from main import draft_name, load_model, speculative_decode, target_name


@torch.inference_mode()
def baseline_generate(model, input_ids, max_new_tokens=100, temperature=1.0):
    """Vanilla greedy/sampling autoregressive decode with KV-cache. Batch 1 only.

    Uses the same explicit mask/position plumbing as `generate` so hybrid
    sliding-window Gemma4 masks stay correct. No `crop()` here, so no
    `activate_past_recording()` (sliding layers truncate naturally).
    """
    assert input_ids.shape[0] == 1
    device = model.device
    eid = model.generation_config.eos_token_id
    if isinstance(eid, int):
        eid = [eid]
    eos_ids = torch.tensor(eid, dtype=torch.long, device=device) if eid is not None else None

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

    seq = input_ids.to(device).clone()
    prompt_len = seq.shape[1]
    shared = None
    out = model(
        input_ids=seq,
        attention_mask=torch.ones(1, prompt_len, dtype=torch.long, device=device),
        position_ids=torch.arange(prompt_len, device=device).unsqueeze(0),
        use_cache=True,
        logits_to_keep=1,
        return_shared_kv_states=True,
    )
    cache = out.past_key_values
    shared = getattr(out, "shared_kv_states", None)
    logits = out.logits[:, -1, :]
    sequence = seq
    generated = 0

    while generated < max_new_tokens:
        nxt = _sample(logits)
        # EOS check
        if eos_ids is not None and torch.isin(nxt[0], eos_ids).any():
            sequence = torch.cat([sequence, nxt], dim=1)
            generated += 1
            break
        sequence = torch.cat([sequence, nxt], dim=1)
        generated += 1
        if generated >= max_new_tokens:
            break
        cur_len = sequence.shape[1] - 1  # position of `nxt`
        kwargs = {
            "input_ids": nxt,
            "attention_mask": torch.ones(1, cur_len + 1, dtype=torch.long, device=device),
            "position_ids": torch.tensor([[cur_len]], dtype=torch.long, device=device),
            "past_key_values": cache,
            "use_cache": True,
            "logits_to_keep": 1,
            "return_shared_kv_states": True,
        }
        if shared is not None:
            kwargs["shared_kv_states"] = shared
        out = model(**kwargs)
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        shared = getattr(out, "shared_kv_states", shared)

    return sequence[:, prompt_len:]


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
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]
        print(f"\n=== prompt ({prompt_len} tokens): {prompt[:80]!r} ===")
        print(f"max_new_tokens={max_new_tokens} draft_tokens={draft_tokens} T={temperature}")

        # warmup (compile caches, etc)
        for _ in range(warmup):
            baseline_generate(target_model, input_ids, max_new_tokens, temperature)
            speculative_decode(target_model, draft_model, input_ids, max_new_tokens, draft_tokens, temperature)
            _sync()

        # baseline (track ACTUAL generated lengths: EOS can stop early)
        b_times = []
        b_lens = []
        b_outs = []
        for _ in range(num_runs):
            b_out_run, dt = timed_run(baseline_generate, target_model, input_ids, max_new_tokens, temperature)
            b_times.append(dt)
            b_lens.append(b_out_run.shape[1])
            b_outs.append(b_out_run)
        b_avg = sum(b_times) / len(b_times)
        b_avg_len = sum(b_lens) / len(b_lens) if b_lens else 0
        b_tps = b_avg_len / b_avg if b_avg > 0 else 0

        # speculative (timed + stats: avg accepted tokens per round)
        s_times = []
        s_stats_runs = []
        s_lens = []
        s_outs = []
        for _ in range(num_runs):
            _sync()
            t0 = time.perf_counter()
            s_out_run, stats = speculative_decode(
                target_model, draft_model, input_ids, max_new_tokens, draft_tokens, temperature, return_stats=True
            )
            _sync()
            t1 = time.perf_counter()
            s_times.append(t1 - t0)
            s_stats_runs.append(stats)
            s_lens.append(s_out_run.shape[1])
            s_outs.append(s_out_run)
        s_avg = sum(s_times) / len(s_times)
        s_avg_len = sum(s_lens) / len(s_lens) if s_lens else 0
        s_tps = s_avg_len / s_avg if s_avg > 0 else 0

        # lossless check: greedy (T==0) spec output must equal baseline exactly
        if temperature == 0.0:
            match = all(
                b.shape[1] == b_outs[0].shape[1] and torch.equal(b.cpu(), b_outs[0].cpu())
                for b in b_outs
            ) and all(
                s.shape[1] == b_outs[0].shape[1] and torch.equal(s.cpu(), b_outs[0].cpu())
                for s in s_outs
            )
            print(f"lossless check (T=0 greedy): {'PASS' if match else 'FAIL'}")
            if not match:
                for i, (b, s) in enumerate(zip(b_outs, s_outs)):
                    bb, ss = b[0].cpu().tolist(), s[0].cpu().tolist()
                    diff = next((j for j, (x, y) in enumerate(zip(bb, ss)) if x != y), None)
                    if b.shape[1] != s.shape[1] or diff is not None:
                        print(f"  run {i}: baseline {b.shape[1]} tok vs spec {s.shape[1]} tok, first diff at pos {diff}")
                        break

        speedup = b_avg / s_avg if s_avg > 0 else float("inf")

        # aggregate acceptance stats over num_runs (accuracy accounts for partial final round)
        avg_accepted = sum(s["avg_accepted"] for s in s_stats_runs) / len(s_stats_runs) if s_stats_runs else 0.0
        avg_committed = sum(s["avg_committed"] for s in s_stats_runs) / len(s_stats_runs) if s_stats_runs else 0.0
        avg_rounds = sum(s["rounds"] for s in s_stats_runs) / len(s_stats_runs) if s_stats_runs else 0.0
        total_accepted = sum(s["total_accepted"] for s in s_stats_runs)
        total_proposed = sum(s["total_proposed"] for s in s_stats_runs)
        acceptance_rate = total_accepted / total_proposed if total_proposed else 0.0

        # one correctness sample (decode) - reuse last stats for preview if sampling temp >0
        b_out = baseline_generate(target_model, input_ids, max_new_tokens, temperature)
        s_out, s_sample_stats = speculative_decode(
            target_model, draft_model, input_ids, max_new_tokens, draft_tokens, temperature, return_stats=True
        )
        print(f"baseline : {b_avg:.3f}s avg over {num_runs} runs | {b_tps:.1f} tok/s | {b_avg_len:.1f} tokens avg")
        print(f"speculative: {s_avg:.3f}s avg over {num_runs} runs | {s_tps:.1f} tok/s | {s_avg_len:.1f} tokens avg")
        print(f"speedup: {speedup:.2f}x")
        print(
            f"acceptance: {avg_accepted:.2f} avg accepted draft tok/round | "
            f"{avg_committed:.2f} avg committed tok/round (incl. bonus/replacement) | "
            f"{acceptance_rate:.1%} acceptance rate | {avg_rounds:.1f} rounds avg over {num_runs} runs"
        )
        # show raw times for variance
        print(f"  baseline times: {[f'{x:.3f}' for x in b_times]}")
        print(f"  spec times    : {[f'{x:.3f}' for x in s_times]}")
        # per-run breakdown
        per_run_accepted = [f"{s['avg_accepted']:.2f}" for s in s_stats_runs]
        per_run_committed = [f"{s['avg_committed']:.2f}" for s in s_stats_runs]
        per_run_rounds = [s['rounds'] for s in s_stats_runs]
        print(f"  per-run accepted: {per_run_accepted}")
        print(f"  per-run committed: {per_run_committed}")
        print(f"  per-run rounds   : {per_run_rounds}")
        print(f"  sample rounds detail: accepted_per_round={s_sample_stats['accepted_per_round'][:20]} committed_per_round={s_sample_stats['committed_per_round'][:20]}")
        # truncated decode preview
        print(f"  baseline preview: {tokenizer.decode(b_out[0][:40], skip_special_tokens=True)[:120]!r}")
        print(f"  spec preview    : {tokenizer.decode(s_out[0][:40], skip_special_tokens=True)[:120]!r}")

        if torch.cuda.is_available():
            print(f"  peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        results.append((prompt, b_avg, s_avg, speedup, avg_accepted, acceptance_rate))

    print("\n=== summary ===")
    for prompt, b_avg, s_avg, speedup, avg_accepted, acceptance_rate in results:
        print(
            f"{speedup:.2f}x | baseline {b_avg:.3f}s | spec {s_avg:.3f}s | "
            f"{avg_accepted:.2f} tok/round ({acceptance_rate:.0%}) | {prompt[:60]!r}"
        )

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
