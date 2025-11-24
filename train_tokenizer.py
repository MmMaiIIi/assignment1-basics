import os
import time
import pickle
import argparse
import psutil

from tests.adapters import run_train_bpe  # 或者从你自己的模块导入

def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes, int]:
    """Return (token_id, token_bytes, length_in_bytes)."""
    tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    return tok_id, tok_bytes, len(tok_bytes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--out_prefix", type=str, required=True)
    parser.add_argument(
        "--special_token",
        type=str,
        default="<|endoftext|>",
        help="Special token to add to vocab.",
    )
    args = parser.parse_args()

    process = psutil.Process(os.getpid())

    print(f"[INFO] Training BPE: input={args.input_path}, "
          f"vocab_size={args.vocab_size}, special={args.special_token}")

    mem_before = process.memory_info().rss / (1024 ** 3)

    t0 = time.perf_counter()
    vocab, merges = run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
    )
    t1 = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 ** 3)

    # 粗略估计峰值内存（更准确可以用 `/usr/bin/time -v`）
    mem_used = max(mem_before, mem_after)

    print(f"[INFO] Training time: {t1 - t0:.2f} seconds")
    print(f"[INFO] Approx. memory in use: {mem_used:.2f} GB")

    # 找最长 token
    tok_id, tok_bytes, tok_len = find_longest_token(vocab)
    try:
        tok_str = tok_bytes.decode("utf-8")
    except UnicodeDecodeError:
        tok_str = tok_bytes.decode("utf-8", errors="replace")

    print(f"[INFO] Longest token id={tok_id}, length={tok_len}, repr={tok_str!r}")

    # 序列化 vocab / merges
    vocab_path = f"{args.out_prefix}_vocab.pkl"
    merges_path = f"{args.out_prefix}_merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"[INFO] Saved vocab to {vocab_path}")
    print(f"[INFO] Saved merges to {merges_path}")

if __name__ == "__main__":
    main()
