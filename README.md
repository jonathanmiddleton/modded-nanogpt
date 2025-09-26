# Daisy-Wee - Efficient GPT for Small‑B Models

Daisy-Wee is a fork of the excellent [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt.git), itself a fork of the venerable [nanoGPT](https://github.com/karpathy/nanoGPT.git):


Purpose:
- Evolve the core GPT architecture to *efficiently* support 1.5B+ parameter models (and beyond) on commodity multi-GPU setups.
- Provide an instruction-following fine-tuning pipeline.
- Explore an MoE (Mixture-of-Experts) variant designed for throughput and memory efficiency on small devices.

Status:
- A 350M-class baseline is available with an efficient training loop and an inference path.
- Architectural components are implemented to support efficient long-context training via sliding-window/block-sparse patterns and rotary embeddings.
- An MoE variant is planned and will be layered on top of the same training/runtime scaffolding.
- There is intentional overlap between model and training code paths right now; this will be refactored to reduce duplication and centralize model definitions.

Highlights:
- Efficient attention with sliding-window and block-sparse masking for long contexts.
- Rotary positional embeddings.
- Fused QKV(+O) projection pattern for speed.
- Learned residual gating/scaling to stabilize deeper networks.
- Parameter grouping and optimizer specialization (including a matrix-preconditioned optimizer) for better convergence and throughput.
- BF16-first training with careful casting at hot spots.
- Data sharding pipeline for instruction-tuning corpora.

-------------------------------------------------------------------------------

## Architecture Overview

This project builds on a vanilla decoder-only Transformer (GPT) with a number of pragmatic, efficiency-oriented enhancements.

Key ideas:
- Token Embeddings and Head
  - Standard learned token embedding table for input tokens.
  - Projection to vocabulary logits at the end of the network.
  - Weight shapes and casting are organized for BF16 inference/training efficiency, while preserving numeric stability in sensitive reductions.

- Rotary Positional Embeddings (RoPE)
  - Rotary embeddings are applied to query/key vectors to inject position information without fixed sin/cos tables per position index.
  - This approach works well with long contexts and sliding-window attention.

- Attention: Fused, Windowed, and Block-Sparse
  - Multi-head self-attention with a fused QKV(+O) projection pattern to lower kernel launches and improve memory locality.
  - Sliding-window attention is used to bound attention complexity to O(T·W) rather than O(T²), where W is the local window. This is extended with block masks to blend short-range dense windows with periodic long-range connectivity.
  - Masks are constructed at runtime to mix local windows with occasional long-distance links, preserving long-context recall while controlling cost.

- Feed-Forward (MLP) and Residual Path Design
  - Standard two-layer MLP with non-linearity, sized for target model scales.
  - Learned scalar gates on residual paths stabilize training and enable “multi-path” residual mixing. This allows selective reuse of intermediate features at deeper layers with tunable influence.

- Depth Skips with Learned Gating
  - Some blocks can receive skip connections from earlier layers, gated by learned scalars. This pattern improves gradient flow and can accelerate convergence in deeper stacks.

- Value Embedding Side Channels (for conditioning)
  - The network supports injecting additional “value-like” embeddings at chosen layers. This is useful for instruction conditioning or task-specific control without altering the core token stream.
  - The mechanism is designed to be lightweight: when not in use, the extra path is effectively a no-op.

- Long-Context via Block Masks
  - A mask generator produces a pair of block masks for “short-range” and “long-range” connectivity.
  - These masks are scheduled across layers to alternate or interleave receptive field patterns, balancing locality with periodic global reach.

- Precision Strategy
  - BF16 is used by default in compute-heavy areas, with care to upcast where necessary for stability (e.g., reductions, logits).
  - The intention is to keep memory footprint small and throughput high on modern GPUs.

-------------------------------------------------------------------------------

## Training System and Optimizations

The training loop focuses on throughput, stability, and memory efficiency:

- Distributed Training
  - Multi-GPU training via distributed data parallelism.
  - Data is sharded per rank with consistent iteration semantics, validation steps, and checkpointing.

- Optimizer Strategy
  - Parameter grouping: embeddings, hidden matrices, scalars/gates, and output head can be split into specialized parameter groups.
  - Dual-optimizer setup: a matrix-preconditioned optimizer (inspired by second-order ideas) can be used for large matrices, while a lightweight first-order optimizer handles small scalar/vector parameters.
  - Matrix preconditioning leverages an approximate inverse square-root via a stable iterative method (e.g., Newton–Schulz-type updates), helping control curvature for large weight matrices without fully-fledged second-order methods.

- Learning Rate Schedule and Warmup/Cooldown
  - Standard warmup followed by a schedule that can include linear decay and cooldown.
  - Designed to maintain stability across depth and when increasing sequence length.

- BF16 and Casting Discipline
  - BF16-first training with casting of activations/weights at key points, minimizing precision-related instabilities without giving up memory savings.

- Attention Backends
  - The attention path is structured so you can swap in optimized kernels (e.g., Flash/Flex-style attention) when available for further speedups.
  - Sliding-window and block masks are computed efficiently; masking is applied in a way compatible with fused kernels.

- Data Pipeline
  - Tokenized datasets are stored in compact binary shards with lightweight headers for fast sequential reads.
  - An instruction mix builder script can generate training and validation shards for instruction-following fine-tuning.
  - Validation shards are small and reused frequently for quick sanity checks during training.

-------------------------------------------------------------------------------

## Instruction-Following Fine-Tuning

- Data Format
  - The corpus is tokenized to 16-bit ID streams with a small header per shard describing length and version.
  - Instruction–response pairs are concatenated with consistent separators and special tokens. This enables straightforward next-token prediction training for instruction following.

- Sharding
  - A helper script is provided to build/refresh shards from raw instruction datasets, write training shards up to a configurable token budget, and create a tiny validation shard.

- Training
  - Fine-tuning runs use the same model/training scaffolding as pretraining, with different sequencing and sampling of instruction data.
  - For small to mid-size models, a modest LR with warmup and conservative batch sizing is recommended to avoid overfitting.

-------------------------------------------------------------------------------

## Roadmap for MoE (Mixture-of-Experts)

Planned features:
- Expert-Parallel FFN
  - Replace dense MLPs with top-k gated experts (per token), keeping attention shared across tokens.
  - Router regularization and load balancing to prevent expert collapse.
- Efficient Expert Placement
  - Expert parallelism across devices with batched dispatch/collection.
  - Cache-friendly layouts and fused combine/scatter steps to minimize overhead.
- Compatibility
  - Same attention/masking stack and precision strategy.
  - Drop-in replacement for the dense MLP in most configurations.

The MoE variant will be introduced incrementally and benchmarked against the dense baseline for quality and throughput.

-------------------------------------------------------------------------------

## Scaling to 1.5B+ Parameters

Tips and guidance:
- Use BF16 on modern GPUs for memory efficiency.
- Prefer sliding-window/block-sparse attention with a window sized to your memory budget; extend global reach by interleaving long-range blocks rather than going fully dense.
- Activation checkpointing can be layered in if you need additional memory headroom.
- Group parameters into specialized optimizer sets (e.g., large matrices vs. scalars) to keep step times predictable and numerically stable.
- Increase sequence length and batch size gradually, monitoring loss curvature and validation perplexity; adjust the long/short block schedule rather than increasing dense attention radius.

-------------------------------------------------------------------------------

## Setup

- Environment
  - A conda environment file and a requirements list are provided. Create and activate an environment using your preferred Conda workflow.
  - Python 3.12 is supported.

- Quickstart
  1) Prepare data shards for instruction fine-tuning.
  2) Launch training for the 350M baseline.
  3) Periodically run validation and sampling to monitor quality.

- Example commands
  - Inspect available training options:
    - python train_gpt_baseline.py --help
    - or: python train_gpt_350MM.py --help
  - Start a training run:
    - python -u train_gpt_350MM.py
  - Generate samples after training:
    - python sample.py

Note: Script names and flags may evolve as refactors land. Use --help for the authoritative set of options.

-------------------------------------------------------------------------------

## Repository Structure (high level)

- Model definition implementing GPT blocks with rotary embeddings, fused attention projections, block masks for sliding windows, and learned residual gating/scaling.
- Training script targeting a ~350M baseline that demonstrates the data/optimizer/schedule setup and the sliding-window attention pipeline.
- Utilities for generation and instruction data sharding.

As refactors land, the model definition will become the single source of truth for architecture, while training scaffolding stays modular.

-------------------------------------------------------------------------------

## Acknowledgements

- Based on and inspired by modded-nanogpt, which itself builds on the nanoGPT lineage.
- Thanks to the open-source community for ongoing work on efficient attention kernels, distributed training, and optimizer research.

-------------------------------------------------------------------------------

## License

This repository is released under the terms of the license in the LICENSE file.
