import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

target_name = "Qwen/Qwen3-4B"
draft_name = "Qwen/Qwen3-0.6B"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model(name):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()

    return model


@torch.inference_mode()
def speculative_decode(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens=100,
    draft_tokens=5,
    temperature=1.0,
):

    assert input_ids.shape[0] == 1, "speculative_decode supports batch size 1 only"
    eos_ids = torch.tensor(
        target_model.generation_config.eos_token_id,
        dtype=torch.long,
        device=input_ids.device,
    )

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
        mask = torch.isin(tokens[0], eos_ids)
        if mask.any():
            keep = int(mask.nonzero()[0].item()) + 1
            sequence = torch.cat([sequence, tokens[:, :keep]], dim=1)
            generated_tokens += keep
            return True
        sequence = torch.cat([sequence, tokens], dim=1)
        generated_tokens += tokens.shape[1]
        return False

    # --- Prefill: both caches cover the prompt ---
    target_out = target_model(input_ids=input_ids, use_cache=True)
    target_cache = target_out.past_key_values
    target_logits = target_out.logits[:, -1, :]

    draft_out = draft_model(input_ids=input_ids, use_cache=True)
    draft_cache = draft_out.past_key_values
    draft_logits = draft_out.logits[:, -1, :]

    sequence = input_ids.clone()
    generated_tokens = 0

    while generated_tokens < max_new_tokens:
        remaining = max_new_tokens - generated_tokens
        n_draft = min(draft_tokens, remaining)
        if n_draft == 0:
            break

        # --- 1. Draft phase: propose tokens with the draft model ---
        proposals = []
        draft_dists = []
        cur_logits = draft_logits

        for j in range(n_draft):
            probs = _probs(cur_logits)
            next_token = _sample(probs)
            proposals.append(next_token)
            draft_dists.append(probs)
            if j < n_draft - 1:
                cur_logits = draft_model(
                    input_ids=next_token,
                    past_key_values=draft_cache,
                ).logits[:, -1, :]

        proposed_tokens = torch.cat(proposals, dim=1)

        # --- 2. Verification phase: score all proposals in one target pass ---
        # logits[:, i] predicts the position right after proposed_tokens[:, i]
        verification_logits = target_model(
            input_ids=proposed_tokens,
            past_key_values=target_cache,
        ).logits

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
        for chunk in to_commit:
            if generated_tokens >= max_new_tokens:
                finished = True
                break
            if _append(chunk):  # handles EOS truncation
                finished = True
                break
        if finished or generated_tokens >= max_new_tokens:
            break

        # --- 4. Cache realignment ---
        if replacement_token is not None:
            if n_draft - n_accepted > 0:
                target_cache.crop(-(n_draft - n_accepted))
            if n_draft - 1 - n_accepted > 0:
                draft_cache.crop(-(n_draft - 1 - n_accepted))

        # feed each cache only the suffix it is still missing
        for cache, model in ((target_cache, target_model), (draft_cache, draft_model)):
            missing = sequence[:, cache.get_seq_length() :]
            if missing.shape[1]:
                logits = model(input_ids=missing, past_key_values=cache).logits[
                    :, -1, :
                ]
                if cache is target_cache:
                    target_logits = logits
                else:
                    draft_logits = logits

    return sequence[:, input_ids.shape[1] :]


def main():
    prompt = "Explain speculative decoding in simple terms."

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    target_model = load_model(target_name)
    draft_model = load_model(draft_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)

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
