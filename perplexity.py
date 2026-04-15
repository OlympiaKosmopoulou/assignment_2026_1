import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_perplexity(text, stride, n_ctx, begin_context_tokens):
    model_name = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, tie_word_embeddings=False)
    model.eval()

    tokens = tokenizer(text).input_ids
    bos_token = tokenizer.bos_token_id

    total_loss = 0.0
    token_count = 0

    for start in range(0, len(tokens), stride):
        end = min(start + n_ctx, len(tokens))

        window = tokens[start:end]

        window = [bos_token] + window

        window_tensor = torch.tensor([window])

        with torch.no_grad():
            logits = model(window_tensor).logits

        if start == 0:
            context_start = begin_context_tokens
        else:
            context_start = n_ctx - stride

        for i in range(context_start, len(window) - 1):
            row = logits[0, i].tolist()

            max_val = max(row)
            shifted = [x - max_val for x in row]

            sum_exp = sum(math.exp(x) for x in shifted)
            log_sum_exp = math.log(sum_exp)

            log_probs = [x - log_sum_exp for x in shifted]

            target_token = window[i + 1]
            log_prob = log_probs[target_token]

            total_loss += -log_prob
            token_count += 1

    avg_nll = total_loss / token_count
    perplexity = math.exp(avg_nll)

    return perplexity

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--begin-context-tokens", type=int, default=512)

    parser.add_argument("input_file")
    parser.add_argument("output_file")

    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    ppl = compute_perplexity(
        text,
        args.stride,
        args.n_ctx,
        args.begin_context_tokens
    )

    with open(args.output_file, "w") as f:
        f.write(f"{ppl}\n")


if __name__ == "__main__":
    main()

