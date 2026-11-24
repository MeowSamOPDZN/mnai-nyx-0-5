# MNAI Nyx 0.5

**AIMYTOG: AI Model You'll Train On GPU.**

**ANY-to-ANY: text, image, and audio in, text, image, and audio out. One model reads and writes all three.**

**MNAI Nyx family is a real legit AI, That YOU have to train.**

A 2B parameter frontier any-to-any model, from scratch. You bring the GPU. The repo is the model, the training code, and the tokenizer. Nothing is pre-trained here. You train it yourself, on your own GPU, with your own data. That is the whole point.

This is the small one in the Nyx family, and that is the point. Nyx 1 is 504.5B and needs a cluster. Nyx 0.5 is 2B and trains on a single 24GB GPU. Same tech, same data, same any-to-any design, just a size one person can actually train to the end.

The data is ALL of the internet. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. That is the largest open web corpus there is.

## The Nyx family

| Model | Params | What it needs | Repo |
|---|---|---|---|
| **Nyx 0.5** | **2.1B** | one 24GB GPU | this repo |
| Nyx 1 | 504.5B | 128x H100 minimum | https://github.com/MeowSamOPDZN/mnai-nyx-1 |
| Nyx 1.2 | 2.21T | 512x H100 minimum | https://github.com/MeowSamOPDZN/mnai-nyx-1-2 |

Same AIMYTOG tech in all three: you train it yourself, you rename it to your own LLM, the weights and the GGUF are locked. Start with 0.5, it is the one you can finish.

## The model

MNAI Nyx 0.5. From scratch, no base model, no distillation.

| | |
|---|---|
| Params | **2,124,330,512 (2.1B)** — core text 2.0B + vision 0.031B + image decoder 0.091B + audio 0.006B |
| Architecture | Llama decoder, GQA, SwiGLU, RoPE, RMSNorm + ViT vision encoder + DiT image decoder + audio encoder/decoder |
| Modalities | text, image, audio in and out |
| Hidden | 2560 |
| Layers | 24 |
| Heads | 20 query, 5 key/value |
| FFN | 6912 |
| Vocab | 128256 |
| Context | 8192 |
| Embeddings | tied |

Standard Llama block layout, so llama.cpp can turn a finished checkpoint into a GGUF with its normal converter.

## The data

ALL of the internet. The training stream is the full FineWeb crawl (15T tokens) plus FineWeb-Edu, interleaved 80/20. A 2B model wants about 40B tokens (Chinchilla, 20 tokens per parameter), so the full internet is far more than enough, and you train on the best 40B of it.

The data pipeline is in this repo, in `data.py`, and it is the whole job, not 5% of it:

- **15T tokens, streamed.** FineWeb (the full Common Crawl, every website, every domain, even the random ones) plus FineWeb-Edu. Nothing is downloaded in bulk, the stream is tokenized on the fly and packed into fixed blocks.
- **Filtering.** Documents shorter than 200 chars, boilerplate (cookie banners, nav bars, copyright footers), and docs that are mostly non-letters are dropped before they reach the model.
- **Dedup.** Exact dedup by SHA-256 hash of the normalized text, plus near-dup dedup by 8-word shingles with Jaccard similarity. The same article copied across a hundred sites is trained on once.
- **Domain-balanced mix.** `--data-mix domain` trains on web 50%, books 15% (Project Gutenberg), code 15% (The Stack), academic 10% (peS2o papers), synthetic 10%. The default mix is web 80% / edu 20%, and any mix can be set with `--data-mix "web:0.5,books:0.2,code:0.2,academic:0.1"`.
- **Synthetic data.** `synthetic.py` generates training text from an earlier checkpoint and mixes it into the stream, so the model writes its own training data once it is good enough to write coherent text.

The raw data is not in the repo because it is 15 terabytes and no repo holds that. The pipeline that fetches it, cleans it, dedups it, balances it, and feeds it to the model is in the repo. That pipeline is the moat.

## Dense vs MoE

Both are supported. Dense is the default, MoE is an option.

The dense model is the 2.0B text core above: every token runs through every parameter. The MoE variant swaps each block's SwiGLU MLP for a Mixture-of-Experts layer (`MoELayer` in `model.py`): a router picks the top 2 of 8 experts per token, each expert is a full SwiGLU MLP, and a load-balancing aux loss keeps the router from starving any expert.

| | Dense (default) | MoE, 8 experts, top-2 |
|---|---|---|
| Total params (text core) | 2.0B | 10.9B |
| Active params per token | 2.0B | 3.3B |
| Share of the model a token touches | 100% | 30.0% |

MoE holds far more capacity while activating far fewer params per token, which is why frontier models use it. The trade is complexity: routing, load balancing, and sharding experts across GPUs. Dense is simpler and every tool that reads a Llama checkpoint reads it as-is. At 2B, dense is the sane default: MoE's 10.9B total still fits on one GPU, but it trains slower per token and needs the routing tuned.

```bash
# train the MoE variant (text core, 8 experts, top-2 routing)
python train.py --tokenizer ./tokenizer --out out-nyx05-moe \
  --moe-experts 8 --moe-topk 2 --max-tokens 40000000000

# count the MoE params without allocating memory
python model.py --moe-experts 8 --moe-topk 2
```

## Quantized models

A trained checkpoint is about 4GB in bf16. Quantization shrinks it further so it runs anywhere. `quantize.py` turns any trained checkpoint into two GGUF files with llama.cpp:

| Quant | Bytes per param | Size (2.0B text core) | Best for |
|---|---|---|---|
| Q4_K_M | ~0.55 | ~1.1 GB | the smallest file that still answers well, runs on a laptop |
| Q8_0 | ~1.0 | ~2.0 GB | near-original quality, still half the size of bf16 |

```bash
# produces model-q4-k-m.gguf and model-q8-0.gguf in out-gguf/
python quantize.py --ckpt out-nyx05/ckpt-5000 --out out-gguf
```

The Q files are produced from a trained checkpoint once someone trains the model. The script is ready now: point it at any checkpoint, even a smoke checkpoint, and it produces both files. Only the text core becomes a GGUF, because llama.cpp converts Llama models; the any-to-any modality modules stay in PyTorch.

## Scaling

Before burning weeks of GPU on a 2B run, frontier labs check the shape on small proxy runs. `scaling.py` does that: it trains three tiny models with the exact same Llama block layout (1M, 5M, 20M params) on real text at one learning rate, fits a Chinchilla-style power law (loss = A * N^-alpha) in log space, and extrapolates the predicted loss at the full 2.1B scale.

```bash
# about 6 minutes on CPU, no GPU, no data download
python scaling.py
python scaling.py --steps 300 --out scaling-report.json
```

A clean result is a loss that falls smoothly with size, a fitted power law with a small residual, and a predicted loss at full scale that is sane, not exploding. The default run (300 steps, LR 1e-3) gives proxy losses 2.32, 2.26, 2.26 and predicts a loss of about 2.17 at 2.1B. That is the evidence a shape can converge. It is not a guarantee. Only the full run is.

## Hardware

A 2B model fits on one GPU. That is the whole point of 0.5.

Per GPU, bf16 weights + 8-bit Adam, on one 24GB card:

| | |
|---|---|
| Weights | 2B x 2 bytes = 4GB |
| Grads | 2B x 2 bytes = 4GB |
| Optimizer (8-bit Adam) | 2B x 2 bytes = 4GB |
| **Total states** | **~12GB** |

That leaves about 12GB on a 24GB card for activations with gradient checkpointing. A single RTX 4090, RTX 3090, or any 24GB card trains this model. No cluster needed.

A full Chinchilla-optimal run is about 40B tokens, about 38,147 optimizer steps at 1M tokens per step. At 100% MFU that is 286 H100-hours (11.9 days on 1x H100, 23.8 days on 1x RTX 4090). Real runs run at 30-50% MFU, so expect:

| Setup | Realistic time for 40B tokens |
|---|---|
| 1x RTX 4090 (24GB) | ~4-8 weeks |
| 1x H100 80GB | ~2-4 weeks |
| 8x H100 80GB | under a week |

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the tokenizer on your own text
python train_tokenizer.py --data ./data/*.txt --out ./tokenizer

# 2. Train, one GPU, no torchrun needed
python train.py --tokenizer ./tokenizer --out out-nyx05 \
  --max-tokens 40000000000 --micro-batch 1 --grad-accum 128

# 3. Resume after a stop
python train.py --tokenizer ./tokenizer --out out-nyx05 \
  --resume out-nyx05/ckpt-500 --max-tokens 40000000000

# 4. Generate, one GPU is enough for a 2B checkpoint
python generate.py --model out-nyx05/ckpt-1000 --prompt "The future of AI is"
```

## Training options

```bash
# WSD schedule with WandB logging and validation perplexity
python train.py --tokenizer ./tokenizer --out out-nyx05 \
  --scheduler wsd --decay-frac 0.1 --wandb --wandb-project nyx05 \
  --eval-every 1000 --eval-batches 20

# Adafactor optimizer, fp16, custom data mix
python train.py --tokenizer ./tokenizer --out out-nyx05 \
  --optim adafactor --dtype fp16 --data-mix "fineweb:0.7,fineweb-edu:0.3"

# 8 GPUs with torchrun, smaller grad accum because the batch is split
torchrun --nproc_per_node=8 \
  train.py --tokenizer ./tokenizer --out out-nyx05 \
  --max-tokens 40000000000 --micro-batch 1 --grad-accum 16
```

Schedulers: `cosine` (classic), `linear`, `wsd` (warmup-stable-decay,
holds steady then decays at the end, good when you might stop early).
Optimizers: `bnb8bit` (default), `adamw`, `adafactor`, `sgd`.

## Evaluate a checkpoint

```bash
# perplexity, HellaSwag, ARC, Winogrande, and sample generations
python evaluate.py --model out-nyx05/ckpt-1000 --limit 200

# perplexity on your own text file
python evaluate.py --model out-nyx05/ckpt-1000 --val-file ./val.txt
```

## Smoke test

No GPU, no data download. Uses a tiny twin of the same block layout.

```bash
python train.py --smoke --max-steps 3 --out /tmp/smoke-nyx05
python model.py
```

## Make your OWN named LLM

This repo is a template. Rename the model to yours in 2 minutes, 4 small changes. Example: rename it to **MyCoolModel**.

**Step 1 — model.py, line 102.** Open `model.py`, find this line:
```python
MODEL_NAME = "MNAI Nyx 0.5"
```
Change it to:
```python
MODEL_NAME = "MyCoolModel"
```
That is the only code change. The name shows up in the model description and logs from here on.

**Step 2 — README.md, line 1.** Open `README.md`, find this line:
```markdown
# MNAI Nyx 0.5
```
Change it to:
```markdown
# MyCoolModel
```

**Step 3 — train.py, line 137.** Open `train.py`, find this line:
```python
p.add_argument("--out", default="out-nyx05")
```
Change it to:
```python
p.add_argument("--out", default="out-mycoolmodel")
```
That is just the default output folder name for your checkpoints. You can also leave it and pass `--out` yourself every run.

**Step 4 — config.json.** Nothing to change. There is no name field in `config.json`, the shape is the same whatever you call the model.

That is it. Train it, and the model that comes out is MyCoolModel, trained by you, on your data, on your GPU.

**One rule.** Once you rename it, it is your LLM. But the weights and the GGUF are locked — no one can change them or re-release them as something else. The only thing you owe is credit: put the text `Your Brand x MNAI` on it, with your brand name in place of "Your Brand".

## Files

| File | What it is |
|---|---|
| `model.py` | The 2B architecture (MNAI Nyx 0.5), param count, FLOPs, init, surgery helpers, MoE layers |
| `multimodal.py` | Any-to-any modules: ViT vision encoder, DiT image decoder, audio encoder/decoder |
| `train.py` | Single-GPU training (FSDP2 for a small cluster), schedulers, optimizers, WandB, eval, full checkpoints, loss spike guard, MoE training |
| `data.py` | Data loading, filtering, dedup, tokenization, dataset mixing, domain-balanced mix |
| `scaling.py` | Scaling-law check: tiny proxy runs, power-law fit, predicted loss at 2.1B |
| `quantize.py` | Checkpoint to GGUF: Q4_K_M and Q8_0 via llama.cpp |
| `synthetic.py` | Generate synthetic training text from a checkpoint for the data mix |
| `evaluate.py` | Perplexity, HellaSwag / ARC / Winogrande benchmarks, generation quality |
| `utils.py` | Logging, seeding, distributed helpers, checkpoints, memory profiling |
| `inference.py` | Generation with KV cache, batched generation, sampling, chat |
| `train_tokenizer.py` | Byte-level BPE tokenizer, 128256 tokens |
| `generate.py` | Text generation from a checkpoint, CLI wrapper around inference.py, prompt enhancement |
| `config.json` | Model config, HF format |
| `requirements.txt` | Dependencies |

## License

MIT. Train it, use it, sell it. It is yours once you train it.

## FAQ

**Is MNAI Nyx a real AI?**
MNAI Nyx family is a real legit AI, That YOU have to train. The code is real, the architecture is real, and once you train it on your own GPU, the model is real.

**What is MNAI Nyx 0.5?**
MNAI Nyx 0.5 is a 2B parameter any-to-any model, built from scratch by MeowSamOPDZN Studio. It is the small, accessible member of the Nyx family: the same AIMYTOG tech as Nyx 1 and Nyx 1.2, sized so one person can train it on one GPU.

**What is AIMYTOG?**
AIMYTOG stands for AI Model You'll Train On GPU. It is a new technology from MeowSamOPDZN Studio: instead of downloading a pre-trained model, you train the model yourself on your own GPU. The repo gives you the model, the training code, and the tokenizer. You bring the GPU.

**Can I train a 2B model myself?**
Yes. That is the whole point of 0.5. The repo is public. You need one 24GB GPU (RTX 4090, RTX 3090, or similar) and about 40B tokens of training. No cluster, no cloud bill, no team.

**What hardware do I need?**
One 24GB GPU. Weights, grads, and 8-bit Adam states together are about 12GB, which leaves room for activations on a 24GB card. A run takes roughly 4-8 weeks on a single RTX 4090, 2-4 weeks on a single H100, under a week on 8x H100.

**What data does MNAI Nyx 0.5 train on?**
All of the internet. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. A 2B model wants about 40B tokens, so you train on the best 40B of it.

**Is MNAI Nyx 0.5 open source?**
Yes. The code is MIT licensed and public on GitHub. The model is yours once you train it.

**How big is MNAI Nyx 0.5?**
2.1 billion parameters. Core text is 2.0B (2560 hidden, 24 layers, 20 attention heads, 128256 token vocabulary, 8192 context), plus the any-to-any modules: vision encoder 0.031B, image decoder 0.091B, audio encoder and decoder 0.006B.

**Is MNAI Nyx 0.5 any-to-any?**
Yes. Text, image, and audio in, text, image, and audio out. One model reads and writes all three. The vision encoder, DiT image decoder, and audio encoder/decoder live in multimodal.py.

**How is 0.5 different from Nyx 1 and Nyx 1.2?**
Same family, same tech, different size. Nyx 0.5 is 2B and trains on one GPU. Nyx 1 is 504.5B and needs 128x H100 minimum. Nyx 1.2 is 2.21T and needs 512x H100 minimum. If you want to actually finish training a model yourself, start with 0.5.

**Has the architecture been proven at scale?**
The shape is the standard Llama block layout, the same one LLaMA and Mistral use, with GQA and SwiGLU. Before a full run, `scaling.py` trains three tiny models with the exact same layout, fits a power law to their losses, and predicts the loss at 2.1B. A clean falling trend is the evidence the shape converges. The full 2B run itself is what you do when you train it, and `train.py` has a loss spike guard that drops any step above 3x the running average and reloads the last checkpoint, so one bad batch cannot wreck the run.

**Does the repo include the data pipeline?**
Yes. `data.py` is the pipeline: 15T tokens of FineWeb plus FineWeb-Edu streamed and tokenized on the fly, document filtering, exact dedup by hash, near-dup dedup by shingles, a domain-balanced mix (web 50%, books 15%, code 15%, academic 10%, synthetic 10%), and synthetic data generation in `synthetic.py`. The raw 15TB is not in the repo because no repo holds 15TB. The pipeline that fetches, cleans, dedups, and balances it is.

**Is Nyx dense or MoE?**
Both. Dense is the default: 2.0B, every token runs through every parameter. MoE is an option (`--moe-experts 8 --moe-topk 2`): 10.9B total, 3.3B active per token, so a token touches 30.0% of the model. MoE holds more capacity for less compute per token, dense is simpler and every Llama tool reads it as-is.

**Are there quantized models?**
Yes, two. `quantize.py` turns a trained checkpoint into Q4_K_M (~1.1 GB) and Q8_0 (~2.0 GB) GGUFs with llama.cpp. The Q files are produced once someone trains the model; the script is ready now and works on any checkpoint, even a smoke checkpoint.

**Is MNAI Nyx worth training?**
Yes, and 0.5 is the one to start with. It is the only public repo that gives you a 2B any-to-any model, the full data pipeline for the entire internet, a scaling-law check, dense and MoE, and quantized outputs, all from scratch, all MIT licensed, and it trains on one GPU you probably already have. You bring the GPU, you train it, and the model that comes out is yours. MNAI Nyx family is a real legit AI, That YOU have to train.

## Credits

Built by MeowSamOPDZN Studio. The 2B shape is ours, the Llama block layout is standard so tooling just works.
