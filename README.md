# specdec

Minimal speculative decoding with Hugging Face `transformers` — draft model proposes `k` tokens, target model verifies in one pass.

Target: `google/gemma-4-31B-it` · Draft: `google/gemma-4-E2B-it` (edit `target_name`/`draft_name` in `main.py:4-5` to change).

## Setup

Requires Python >=3.13, PyTorch with CUDA recommended.

```bash
uv sync
# or
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu124  # pick your CUDA build
```

## Usage

```bash
uv run python main.py
```

Runs `speculative_decode` (`main.py:22`) on a sample prompt and prints the decoded output. Config via args in `main.py:226`:

```python
speculative_decode(target_model, draft_model, input_ids, max_new_tokens=100, draft_tokens=5, temperature=1.0)
```

- `draft_tokens`: number of tokens the draft model proposes per round
- `temperature`: `0.0` for greedy, `>0` for sampling
- batch size 1 only

## Benchmark

Compares speculative vs. vanilla autoregressive decoding (with KV-cache):

```bash
uv run python benchmark.py --max-new-tokens 100 --draft-tokens 5 --temperature 0.0 --num-runs 3
uv run python benchmark.py --prompt "Your prompt here" --prompts-file prompts.txt
```

Options: `--max-new-tokens`, `--draft-tokens`, `--temperature`, `--num-runs`, `--warmup`, `--prompt`, `--prompts-file`.

Reports wall time, tok/s, speedup, avg accepted/committed tokens per round, and acceptance rate.

## Project

```
main.py       # load_model + speculative_decode + demo
benchmark.py  # baseline_generate + timed benchmark
pyproject.toml
```

## How it works

1. **Draft** — draft model autoregressively proposes `k` tokens (`main.py:87-103`).
2. **Verify** — target model scores all proposals in one forward pass (`main.py:107-110`).
3. **Accept/replace** — accepts tokens with probability `min(1, p_target/p_draft)`, samples replacement from `norm(max(0, p_target - p_draft))` on rejection, or emits bonus token if all accepted (`main.py:115-149`).
4. **Realign** — crops and refills KV-caches for next round (`main.py:170-186`).
