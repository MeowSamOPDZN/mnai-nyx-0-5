"""Generate text with a trained Nyx 0.5 checkpoint.

A thin command line wrapper around inference.py. It loads the
model, runs your prompt through it, and prints the result. For
chat, batched prompts, or streaming output, use inference.py
directly or import it into your own script.

Usage:
  python generate.py --model out-nyx05/ckpt-1000 --prompt "The future of AI is"
  python generate.py --model out-nyx05/ckpt-1000 --prompt "Once upon a time" --max-new 200 --temp 0.8
  python generate.py --model out-nyx05/ckpt-1000 --prompt "Hello" --n 4 --temp 1.0
  python generate.py --model out-nyx05/ckpt-1000 --prompt "Hello" --greedy
  python generate.py --model out-nyx05/ckpt-1000 --chat --prompt "What is the capital of France?"

A 2B checkpoint is about 4GB in bf16. It fits on one 24GB GPU
with room for the KV cache, so this script just loads it and
generates. No sharding, no special engine, one GPU is enough.
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import benchmark_generation, chat, generate, generate_batch


def enhance_prompt(prompt):
    # turn a short prompt into a fuller instruction, so the model
    # writes a fuller answer. Keeps the user's words, adds a little
    # framing around them. If the prompt is already long, leave it
    # alone. Works in any language: non-English prompts are left
    # exactly as they are, no English framing mixed in.
    p = prompt.strip()
    if len(p.split()) >= 12:
        return p
    if not p.isascii():
        return p
    low = p.lower()
    if low.startswith(("write", "explain", "tell", "describe", "what",
                       "why", "how", "when", "where", "who", "once",
                       "the", "a photo", "a picture")):
        return p
    return f"Write a detailed passage about {p}."


def maybe_enhance(prompt, args):
    """Enhance the prompt unless --no-enhance was passed.

    Prints the enhanced prompt when it changed, so you can see
    what the model actually got.
    """
    if args.no_enhance:
        return prompt
    enhanced = enhance_prompt(prompt)
    if enhanced != prompt:
        print(f"enhanced prompt: {enhanced!r}", flush=True)
    return enhanced


def load_model(path, device, dtype=torch.bfloat16):
    """Load a checkpoint and its tokenizer, print what we got."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {path} ({n_params:,} params, {n_params / 1e9:.1f}B)", flush=True)
    return model, tok


def run_one(model, tok, args, device):
    """One prompt, one completion, printed to stdout."""
    if args.greedy:
        args.temp = 0.0
    args.prompt = maybe_enhance(args.prompt, args)
    text = generate(
        model, tok, args.prompt, max_new=args.max_new,
        temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_strings=args.stop, device=device)
    print(text, flush=True)
    return text


def run_many(model, tok, args, device):
    """Same prompt N times, for sampling variety."""
    if args.greedy:
        args.temp = 0.0
    args.prompt = maybe_enhance(args.prompt, args)
    results = []
    for i in range(args.n):
        text = generate(
            model, tok, args.prompt, max_new=args.max_new,
            temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_strings=args.stop, device=device)
        print(f"--- sample {i + 1}/{args.n} ---")
        print(text, flush=True)
        results.append(text)
    return results


def run_batch(model, tok, args, device):
    """Several prompts at once, from --prompts or a file."""
    prompts = [maybe_enhance(p, args) for p in (args.prompts or [])]
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts += [l.strip() for l in f if l.strip()]
    if not prompts:
        prompts = [maybe_enhance(args.prompt, args)]
    results = generate_batch(
        model, tok, prompts, max_new=args.max_new,
        temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_strings=args.stop, device=device)
    for prompt, text in zip(prompts, results):
        print(f"--- prompt: {prompt!r} ---")
        print(text, flush=True)
    return results


def run_chat(model, tok, args, device):
    """One chat turn, user message in --prompt."""
    args.prompt = maybe_enhance(args.prompt, args)
    messages = [{"role": "user", "content": args.prompt}]
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})
    text = chat(model, tok, messages, max_new=args.max_new,
                temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_strings=args.stop, device=device)
    print(text, flush=True)
    return text


def run_bench(model, tok, args, device):
    """Tokens per second, so you know how fast the setup is."""
    res = benchmark_generation(model, tok, prompt=args.prompt,
                               max_new=args.max_new, device=device)
    print(f"tokens per second: {res['tokens_per_second']:.1f}")
    print(f"ms per token: {res['ms_per_token']:.1f}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--prompts", nargs="*", default=None,
                   help="several prompts to run as one batch")
    p.add_argument("--prompts-file", default=None,
                   help="file with one prompt per line, run as a batch")
    p.add_argument("--max-new", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true",
                   help="always pick the most likely token, no sampling")
    p.add_argument("--no-enhance", action="store_true",
                   help="use the prompt exactly as given, no enhancement")
    p.add_argument("--n", type=int, default=1,
                   help="how many samples of the same prompt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stop", nargs="*", default=None,
                   help="stop strings, generation ends at the first one")
    p.add_argument("--chat", action="store_true",
                   help="treat --prompt as one chat turn")
    p.add_argument("--system", default=None,
                   help="system message for --chat")
    p.add_argument("--bench", action="store_true",
                   help="measure tokens per second instead of generating")
    p.add_argument("--out", default=None,
                   help="write the result as JSON to this file")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    model, tok = load_model(args.model, device, dtype=dtype)

    t0 = time.time()
    if args.bench:
        result = run_bench(model, tok, args, device)
    elif args.chat:
        result = run_chat(model, tok, args, device)
    elif args.prompts or args.prompts_file:
        result = run_batch(model, tok, args, device)
    elif args.n > 1:
        result = run_many(model, tok, args, device)
    else:
        result = run_one(model, tok, args, device)
    el = time.time() - t0
    print(f"({el:.1f}s)", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"prompt": args.prompt, "result": result,
                       "seconds": el}, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
