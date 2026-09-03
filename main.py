import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

target_name = "google/gemma-4-31B-it"
draft_name = "google/gemma-4-E2B-it"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model(name):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()

    return model


def _eos_ids_for(model, device):
    """Return EOS ids as a 1-D long tensor (handles int | list | None)."""
    eid = model.generation_config.eos_token_id
    if eid is None:
        return None
    if isinstance(eid, int):
        eid = [eid]
    return torch.tensor(eid, dtype=torch.long, device=device)


def _trim_shared_kv(shared, keep_len):
    """Trim shared_kv_states (Gemma4 KV-sharing / MTP) back to `keep_len` on dim 2."""
    if shared is None:
        return None
    trimmed = {}
    for k, v in shared.items():
        kk, vv = v
        if kk.shape[-2] > keep_len:
            kk = kk[:, :, :keep_len, :]
            vv = vv[:, :, :keep_len, :]
        trimmed[k] = (kk, vv)
    return trimmed


def _forward(model, input_ids, attention_mask, position_ids, cache, shared, logits_to_keep):
    """Single forward with the explicit args HF `generate` uses for Gemma4.

    Always passes `use_cache=True`, explicit 2-D `attention_mask` + `position_ids`,
    and threads `shared_kv_states` when the model supports it (harmless otherwise).
    """
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": cache,
        "use_cache": True,
        "logits_to_keep": logits_to_keep,
    }
    if shared is not None:
        kwargs["shared_kv_states"] = shared
    # Request shared states back; models without support simply ignore / return None.
    kwargs["return_shared_kv_states"] = True
    out = model(**kwargs)
    return out


@torch.inference_mode()
def speculative_decode(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens=100,
    draft_tokens=5,
    temperature=1.0,
    return_stats: bool = False,
):

    assert input_ids.shape[0] == 1, "speculative_decode supports batch size 1 only"
    t_dev = target_model.device
    d_dev = draft_model.device
    eos_ids = _eos_ids_for(target_model, t_dev)

    def _probs(logits):
        if temperature == 0.0:
            probs = torch.zeros_like(logits)
            probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            return probs
        return torch.softmax(logits / temperature, dim=-1)

    def _sample(probs):
        if temperature == 0.0:
            return probs.argmax(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    def _append(tokens):
        nonlocal sequence, generated_tokens
        # `tokens` lives on the target device; `eos_ids` too (None if model has none).
        if eos_ids is not None:
            mask = torch.isin(tokens[0], eos_ids)
            if mask.any():
                keep = int(mask.nonzero()[0].item()) + 1
                sequence = torch.cat([sequence, tokens[:, :keep]], dim=1)
                generated_tokens += keep
                return True
        sequence = torch.cat([sequence, tokens], dim=1)
        generated_tokens += tokens.shape[1]
        return False

    def _ones(n, device):
        return torch.ones(1, n, dtype=torch.long, device=device)

    def _positions(start, length, device):
        return (torch.arange(length, device=device) + start).unsqueeze(0)

    # --- Prefill: both caches cover the prompt (explicit mask/positions) ---
    # Canonical sequence lives on the target device; slices are moved per model.
    prompt_len = input_ids.shape[1]
    sequence = input_ids.to(t_dev).clone()
    target_out = _forward(
        target_model,
        sequence,
        _ones(prompt_len, t_dev),
        _positions(0, prompt_len, t_dev),
        None,
        None,
        1,
    )
    target_cache = target_out.past_key_values
    target_shared = getattr(target_out, "shared_kv_states", None)
    target_logits = target_out.logits[:, -1, :]

    draft_out = _forward(
        draft_model,
        sequence.to(d_dev),
        _ones(prompt_len, d_dev),
        _positions(0, prompt_len, d_dev),
        None,
        None,
        1,
    )
    draft_cache = draft_out.past_key_values
    draft_shared = getattr(draft_out, "shared_kv_states", None)
    draft_logits = draft_out.logits[:, -1, :]

    # Gemma4 uses hybrid sliding-window layers: without this, `crop()` raises
    # once the window (512) is reached, and rollback loses prefix tokens.
    target_cache.activate_past_recording()
    draft_cache.activate_past_recording()

    generated_tokens = 0

    # --- Stats tracking for benchmark: average accepted tokens per round ---
    rounds = 0
    accepted_per_round: list[int] = []
    committed_per_round: list[int] = []
    proposed_per_round: list[int] = []

    while generated_tokens < max_new_tokens:
        remaining = max_new_tokens - generated_tokens
        n_draft = min(draft_tokens, remaining)
        if n_draft == 0:
            break

        # Round-start length == logical cache length (tracked explicitly so we
        # never rely on `cache.get_seq_length()`, which is ambiguous for hybrid
        # sliding/full caches).
        L = sequence.shape[1]

        # --- 1. Draft phase: propose tokens with the draft model ---
        proposals_d = []
        draft_dists_d = []
        cur_logits = draft_logits

        for j in range(n_draft):
            probs = _probs(cur_logits)
            next_token = _sample(probs)
            proposals_d.append(next_token)
            draft_dists_d.append(probs)
            if j < n_draft - 1:
                out = _forward(
                    draft_model,
                    next_token,
                    _ones(L + j + 1, d_dev),
                    _positions(L + j, 1, d_dev),
                    draft_cache,
                    draft_shared,
                    1,
                )
                draft_cache = out.past_key_values
                draft_shared = getattr(out, "shared_kv_states", draft_shared)
                cur_logits = out.logits[:, -1, :]

        proposed_d = torch.cat(proposals_d, dim=1)
        proposed_tokens = proposed_d.to(t_dev)
        # Move draft dists to the target device for the acceptance test.
        draft_dists = [p.to(t_dev) for p in draft_dists_d]

        # --- 2. Verification phase: score all proposals in one target pass ---
        # logits[:, i] predicts the position right after proposed_tokens[:, i]
        v_out = _forward(
            target_model,
            proposed_tokens,
            _ones(L + n_draft, t_dev),
            _positions(L, n_draft, t_dev),
            target_cache,
            target_shared,
            n_draft,
        )
        target_cache = v_out.past_key_values
        target_shared = getattr(v_out, "shared_kv_states", target_shared)
        verification_logits = v_out.logits

        n_accepted = 0
        replacement_token = None

        for i in range(n_draft):
            prior_logits = target_logits if i == 0 else verification_logits[:, i - 1, :]
            target_probs = _probs(prior_logits)
            draft_probs = draft_dists[i]
            draft_token = proposed_tokens[:, i]

            target_prob = target_probs.gather(1, draft_token.unsqueeze(-1)).squeeze(-1)
            draft_prob = draft_probs.gather(1, draft_token.unsqueeze(-1)).squeeze(-1)

            acceptance_ratio = torch.clamp(target_prob / (draft_prob + 1e-9), max=1.0)

            rand = torch.rand_like(acceptance_ratio)
            accepted = rand < acceptance_ratio

            if accepted.all():
                n_accepted += 1
            else:
                adjusted = torch.clamp(target_probs - draft_probs, min=0.0)
                adjusted_sum = adjusted.sum(dim=-1, keepdim=True)
                fallback_mask = adjusted_sum.squeeze(-1) == 0
                if fallback_mask.any():
                    adjusted[fallback_mask] = target_probs[fallback_mask]
                adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-9)

                replacement_token = torch.multinomial(adjusted, num_samples=1)
                break

        # --- 3. Commit accepted tokens + (replacement or bonus) ---
        to_commit = []
        if n_accepted > 0:
            to_commit.append(proposed_tokens[:, :n_accepted])
        if replacement_token is not None:
            to_commit.append(replacement_token)
        elif n_accepted == n_draft:  # fully accepted -> bonus token for free
            to_commit.append(_sample(_probs(verification_logits[:, n_draft - 1, :])))

        finished = False
        committed_before = generated_tokens
        for chunk in to_commit:
            if generated_tokens >= max_new_tokens:
                finished = True
                break
            if _append(chunk):  # handles EOS truncation
                finished = True
                break
        committed_this_round = generated_tokens - committed_before
        # track per-round stats (lightweight)
        rounds += 1
        accepted_per_round.append(n_accepted)
        committed_per_round.append(committed_this_round)
        proposed_per_round.append(n_draft)
        if finished or generated_tokens >= max_new_tokens:
            break

        # --- 4. Cache realignment (Gemma4-aware) ---
        # Verification polluted the target cache with all n_draft tokens and the
        # draft cache with the first n_draft-1. Roll back the rejected suffix.
        # `crop(0)` is NOT a no-op for sliding layers: it shrinks them back to
        # the window, so call it even on full acceptance.
        if replacement_token is not None:
            target_cache.crop(-(n_draft - n_accepted))
            draft_cache.crop(-(n_draft - 1 - n_accepted))
            # Keep any threaded shared-KV in sync (needed for MTP-style
            # assistants; harmless no-op otherwise as it is rebuilt from cache).
            keep = L + n_accepted
            target_shared = _trim_shared_kv(target_shared, keep)
            draft_shared = _trim_shared_kv(draft_shared, keep)
        else:
            target_cache.crop(0)
            draft_cache.crop(0)

        # Logical cache lengths after the crop (derived from our own counters,
        # not from per-layer `get_seq_length()`).
        t_cache_len = L + n_accepted
        if replacement_token is not None:
            d_cache_len = L + n_accepted
        else:
            d_cache_len = L + n_accepted - 1  # last proposal was never fed
        seq_len = sequence.shape[1]

        # Feed each cache only the suffix it is still missing (replacement or
        # bonus for target; replacement or last-proposal+bonus for draft).
        t_missing = sequence[:, t_cache_len:seq_len]
        if t_missing.shape[1]:
            out = _forward(
                target_model,
                t_missing,
                _ones(seq_len, t_dev),
                _positions(t_cache_len, seq_len - t_cache_len, t_dev),
                target_cache,
                target_shared,
                1,
            )
            target_cache = out.past_key_values
            target_shared = getattr(out, "shared_kv_states", target_shared)
            target_logits = out.logits[:, -1, :]

        d_missing = sequence.to(d_dev)[:, d_cache_len:seq_len]
        if d_missing.shape[1]:
            out = _forward(
                draft_model,
                d_missing,
                _ones(seq_len, d_dev),
                _positions(d_cache_len, seq_len - d_cache_len, d_dev),
                draft_cache,
                draft_shared,
                1,
            )
            draft_cache = out.past_key_values
            draft_shared = getattr(out, "shared_kv_states", draft_shared)
            draft_logits = out.logits[:, -1, :]

    output = sequence[:, input_ids.shape[1] :]

    if return_stats:
        avg_accepted = sum(accepted_per_round) / len(accepted_per_round) if accepted_per_round else 0.0
        avg_committed = sum(committed_per_round) / len(committed_per_round) if committed_per_round else 0.0
        total_accepted = sum(accepted_per_round)
        total_proposed = sum(proposed_per_round)
        total_committed = sum(committed_per_round)
        # accurate acceptance rate accounts for final partial round (n_draft < draft_tokens)
        acceptance_rate = total_accepted / total_proposed if total_proposed else 0.0
        stats = {
            "rounds": rounds,
            "accepted_per_round": accepted_per_round,
            "committed_per_round": committed_per_round,
            "proposed_per_round": proposed_per_round,
            "avg_accepted": avg_accepted,
            "avg_committed": avg_committed,
            "total_accepted": total_accepted,
            "total_proposed": total_proposed,
            "total_committed": total_committed,
            "total_generated": generated_tokens,
            "draft_tokens": draft_tokens,
            "acceptance_rate": acceptance_rate,
        }
        return output, stats

    return output


def main():
    prompt = "Explain speculative decoding in simple terms."

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    target_model = load_model(target_name)
    draft_model = load_model(draft_name)

    # Gemma-4 IT models expect a chat template; fall back to raw prompt otherwise.
    try:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(target_model.device)

    output_ids = speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        input_ids=input_ids,
        max_new_tokens=100,
        draft_tokens=5,
    )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
