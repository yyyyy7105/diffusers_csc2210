# Attention Profiling & Masking Experiment Plan for Flux 2 Klein-4B

## 1. Overview

This document describes a structured plan to **profile** and **optimize** the attention mechanism in the Flux 2 Klein-4B transformer architecture. The model has two block types:

| Block Type | Class | Count | Attention Mechanism |
|---|---|---|---|
| **Double-stream** | `Flux2TransformerBlock` | 5 | Joint attention (`Flux2Attention`) — image and text streams are projected separately, concatenated along the sequence dimension, and attend jointly |
| **Single-stream** | `Flux2SingleTransformerBlock` | 20 | Parallel self-attention (`Flux2ParallelSelfAttention`) — image and text tokens are concatenated before the block and processed as a single unified sequence |

**Key configuration (Klein-4B):**
- `num_attention_heads`: 24
- `attention_head_dim`: 128
- Inner dimension: 3072
- RoPE axes: (32, 32, 32, 32)

The attention in both block types uses **QK-RMSNorm**, **Rotary Position Embeddings (RoPE)**, and dispatches to `scaled_dot_product_attention` via `dispatch_attention_fn`.

---

## 2. Phase 1 — Attention Profiling

### 2.1 Goals

1. Extract raw attention weight matrices (Q·Kᵀ / √d) from every attention layer at selected denoising timesteps.
2. Visualize attention heatmaps to identify exploitable structure (locality, sparsity, block-diagonal patterns, text-image cross-attention patterns).
3. Quantify attention entropy and sparsity metrics per block, per head, and across timesteps.

### 2.2 Profiling Methodology

#### 2.2.1 Hook-Based Attention Weight Capture

Since `dispatch_attention_fn` uses `F.scaled_dot_product_attention` (which does not return attention weights by default), we need an alternative extraction strategy.

**Approach: Custom Attention Processor with Explicit Weight Computation**

Create profiling variants of the two attention processors that compute attention weights explicitly for capture while using the efficient SDPA path for the actual forward computation:

```
Flux2AttnProcessor → Flux2ProfilingAttnProcessor
Flux2ParallelSelfAttnProcessor → Flux2ProfilingSelfAttnProcessor
```

Each profiling processor will:
1. Compute `attn_weights = softmax(Q · Kᵀ / √d)` explicitly for recording (detached, no grad).
2. Store the attention weight tensor in a shared dictionary keyed by `(block_type, block_index, timestep)`.
3. Still dispatch the actual computation through the efficient `dispatch_attention_fn` path so that the model output is unaffected.

**Implementation location:** `src/diffusers/models/transformers/transformer_flux2.py` — add new processor classes alongside existing ones.

**Activation:** Add a `set_profiling_mode(enabled: bool)` method to `Flux2Transformer2DModel` that swaps processors between standard and profiling variants.

#### 2.2.2 What to Capture

For each attention layer at each sampled timestep:

| Metric | Description | Shape |
|---|---|---|
| **Raw attention weights** | `softmax(QKᵀ / √d)` | `[B, H, S, S]` |
| **Per-head entropy** | `−Σ p log p` over the key dimension | `[B, H, S]` |
| **Per-head sparsity** | Fraction of weights below threshold ε | Scalar per head |
| **Top-k concentration** | Fraction of total attention captured by top-k keys | `[B, H, S]` |
| **Block-structure score** | Ratio of within-block vs. cross-block attention (text↔text, img↔img, text↔img) | Scalars |

#### 2.2.3 Sampling Strategy

To keep memory and compute manageable:

- **Timesteps:** Sample 5–8 representative normalized timesteps (range [0, 1], where 0 = clean and 1 = full noise) across the denoising schedule (e.g., t ∈ {0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95}). These values are chosen to cover early denoising (high noise, t ≈ 0.95), mid denoising, and late denoising (low noise, t ≈ 0.05) roughly uniformly.
- **Batching:** Profile with batch size 1.
- **Resolution:** Use the model's standard inference resolution (e.g., 512×512 or 1024×1024). Note that image token count = H×W / patch_size².
- **Heads:** Capture all 24 heads; aggregate statistics but also examine per-head variability.
- **Prompts:** Use 3–5 diverse prompts (simple object, complex scene, abstract concept) to check if patterns are prompt-dependent.

#### 2.2.4 Profiling Locations

**A. Double-stream blocks (5 blocks) — Joint Attention (`Flux2Attention`)**

The joint attention concatenates text and image tokens along the sequence dimension:
```
Q, K, V = [text_Q; img_Q], [text_K; img_K], [text_V; img_V]
```

The resulting attention matrix has four quadrants of interest:
```
         text_K    img_K
text_Q [ T→T      T→I   ]
img_Q  [ I→T      I→I   ]
```

Profile each quadrant separately to understand cross-modal attention patterns.

**B. Single-stream blocks (20 blocks) — Self-Attention (`Flux2ParallelSelfAttention`)**

Text and image tokens are already concatenated before entering these blocks. The attention matrix has the same four-quadrant structure, but the tokens share a single processing pathway.

### 2.3 Visualization Plan

1. **Full attention heatmaps:** Per-head, per-block, per-timestep attention weight matrices rendered as 2D heatmaps (use `matplotlib.imshow`). Subsample or average across heads for overview plots.
2. **Quadrant-level summaries:** For each block, compute mean attention weight in each of the four quadrants (T→T, T→I, I→T, I→I) and plot as a bar chart across blocks.
3. **Sparsity profiles:** Plot per-block attention entropy and sparsity as line charts across block depth.
4. **Temporal evolution:** Plot how attention sparsity and quadrant ratios change across denoising timesteps.
5. **Spatial locality maps:** For I→I attention, check if tokens attend preferentially to spatially nearby tokens by computing correlation between attention weight and spatial distance.

### 2.4 Expected Patterns to Investigate

- **Text-image decoupling in later blocks:** Do later single-stream blocks show diminishing cross-modal attention?
- **Spatial locality in image self-attention:** Do image tokens primarily attend to local neighbors?
- **Attention head specialization:** Do some heads specialize in local vs. global patterns?
- **Timestep dependence:** Is attention sparser at low-noise timesteps (late denoising) vs. high-noise (early)?
- **Redundancy across heads:** Do multiple heads learn similar attention patterns?

---

## 3. Phase 2 — Attention Masking for Efficiency

### 3.1 Goals

1. Design attention masks that zero out or skip low-value attention computations.
2. Measure the trade-off between mask aggressiveness and output quality.
3. Identify which blocks and timesteps are most amenable to masking.

### 3.2 Masking Strategies

#### Strategy A: Static Quadrant Masking (Double-stream blocks)

Based on Phase 1's quadrant analysis, apply a static binary mask that blocks specific quadrant interactions:

| Variant | Mask Description | Rationale |
|---|---|---|
| A1 | Block T→T in later double-stream blocks | If text self-attention is redundant after initial layers |
| A2 | Block I→I cross-attention (keep only T↔I) | If image self-attention is handled by single-stream blocks |
| A3 | Block T→I in later double-stream blocks | If text←image feedback diminishes |

**Implementation:** Pass `attention_mask` tensor to `dispatch_attention_fn`. Construct the mask based on text/image token boundaries (known from `encoder_hidden_states.shape[1]`).

#### Strategy B: Spatial Locality Masking (Single-stream blocks, I→I region)

If Phase 1 reveals spatial locality in image self-attention:

| Variant | Mask Description | Rationale |
|---|---|---|
| B1 | Local window attention for I→I | Each image token attends only to tokens within a spatial window of radius r |
| B2 | Dilated/strided local attention | Attend to every k-th token in a larger window for multi-scale coverage |
| B3 | Hybrid: local for I→I, full for T↔I | Preserve full cross-modal attention while sparsifying image self-attention |

**Implementation:** Compute the 2D spatial position of each image token from `img_ids` (which contain height, width coordinates). Build a distance-based mask.

#### Strategy C: Top-k / Threshold Masking (Adaptive)

Apply a data-dependent mask based on attention magnitude:

| Variant | Mask Description | Rationale |
|---|---|---|
| C1 | Top-k attention: each query only attends to its top-k keys | Prune low-weight tail |
| C2 | Threshold mask: zero out weights below ε | Remove negligible interactions |
| C3 | Approximate top-k using locality-sensitive hashing (LSH) | Efficient approximate sparse attention |

**Note:** Strategies C1 and C2 require a two-pass approach or approximation and may not integrate cleanly with `F.scaled_dot_product_attention`. Consider using custom CUDA kernels or xformers block-sparse attention.

#### Strategy D: Block-Depth-Selective Masking

Apply masking only to selected block indices based on profiling insights:

| Variant | Description |
|---|---|
| D1 | Mask only the last N single-stream blocks | If later blocks show high sparsity |
| D2 | Mask only early double-stream blocks | If early blocks show low cross-modal attention |
| D3 | Progressive masking: increasingly aggressive masks in deeper blocks | Gradually reduce computation |

### 3.3 Implementation Architecture

```
Flux2Transformer2DModel
├── set_attention_mask_config(config: AttentionMaskConfig)
│   Stores mask configuration (strategy, block indices, parameters)
│
├── _build_attention_mask(block_type, block_index, text_seq_len, img_seq_len, timestep)
│   Constructs the attention mask tensor based on config
│   Returns: [1, 1, S, S] boolean mask (True = attend, False = skip)
│
└── forward()
    ├── Double-stream loop: pass block-specific mask to each Flux2TransformerBlock
    └── Single-stream loop: pass block-specific mask to each Flux2SingleTransformerBlock
```

**Mask passing mechanism:** The existing `attention_mask` parameter is already supported in both `Flux2AttnProcessor.__call__` and `Flux2ParallelSelfAttnProcessor.__call__`, and it flows through to `dispatch_attention_fn`. No changes to the attention core are needed — only mask construction and injection at the block level.

### 3.4 Experimental Matrix

#### Independent Variables

| Axis | Values | Total Levels |
|---|---|---|
| **Block type** | Double-stream (joint attn), Single-stream (self-attn) | 2 |
| **Masking strategy** | A (quadrant), B (spatial locality), C (top-k/threshold), D (depth-selective) | 4 |
| **Block depth** | All blocks, Early only (blocks 0–1 / 0–6), Mid only (blocks 2–3 / 7–13), Late only (blocks 3–4 / 14–19) | 4 |
| **Mask aggressiveness** | Conservative (≤10% pruned), Moderate (10–30%), Aggressive (30–50%), Extreme (>50%) | 4 |
| **Timestep dependence** | Fixed mask (all timesteps), Timestep-adaptive (different masks for early/mid/late denoising) | 2 |

#### Priority Experiments (Recommended Execution Order)

| Priority | Experiment | Strategy | Block Type | Depth | Aggressiveness |
|---|---|---|---|---|---|
| P0 | Baseline | None | All | All | 0% |
| P1 | Quadrant masking: block T→T in late double-stream | A1 | Double | Late (3–4) | Moderate |
| P2 | Spatial locality: local window r=8 for I→I in all single-stream | B1 | Single | All | Moderate |
| P3 | Depth-selective: mask last 5 single-stream blocks | D1 | Single | Late (15–19) | Moderate |
| P4 | Combined: P1 + P2 | A1+B1 | Both | Mixed | Moderate |
| P5 | Aggressiveness sweep on best strategy from P1–P4 | Best | Best | Best | Conservative→Extreme |
| P6 | Timestep-adaptive: relax masks at high-noise timesteps | Best+adaptive | Best | Best | Varies |

### 3.5 Evaluation Criteria

#### Quality Metrics

| Metric | Description | Tool |
|---|---|---|
| **FID** (Fréchet Inception Distance) | Distribution-level image quality | `torchmetrics` or `clean-fid` |
| **CLIP Score** | Text-image alignment | OpenCLIP |
| **LPIPS** | Perceptual similarity to unmasked baseline | `lpips` library |
| **SSIM** | Structural similarity to baseline | `torchmetrics` |
| **Visual inspection** | Side-by-side comparison grid | Manual / notebook |

#### Efficiency Metrics

| Metric | Description | Measurement |
|---|---|---|
| **Wall-clock time** (per image) | End-to-end generation latency | `time.perf_counter` or existing profiler |
| **Attention FLOP reduction** | Theoretical FLOP savings from masking (fraction of attention FLOPs eliminated) | Compute from mask sparsity ratio: `FLOP_savings = 1 − (non-masked entries / total entries)` |
| **Peak GPU memory** | Maximum GPU memory during generation | `torch.cuda.max_memory_allocated` |
| **Attention kernel time** | Time spent in attention ops per block | `torch.profiler` with `record_function` labels |

#### Acceptance Criteria

| Threshold | FID Δ | CLIP Score Δ | LPIPS vs. Baseline | Speedup |
|---|---|---|---|---|
| **Acceptable** | < +5% | < −2% | < 0.05 | > 1.1× |
| **Good** | < +2% | < −1% | < 0.02 | > 1.2× |
| **Excellent** | < +1% | ≈ 0 | < 0.01 | > 1.3× |

---

## 4. Implementation Roadmap

### Step 1: Build Profiling Infrastructure (Phase 1)

1. **Create `Flux2ProfilingAttnProcessor` and `Flux2ProfilingSelfAttnProcessor`** in `transformer_flux2.py`.
   - Override `__call__` to compute and store attention weights alongside the normal forward pass.
   - Store weights in a global or model-attached dictionary.

2. **Add `set_profiling_mode()` to `Flux2Transformer2DModel`**.
   - Swap between standard and profiling processors on all attention modules.

3. **Create a profiling notebook** (`2210/attention_profiling.ipynb`).
   - Load the Klein-4B model.
   - Run inference with profiling enabled on representative prompts.
   - Generate heatmaps, entropy plots, and quadrant analyses.
   - Save raw attention data for offline analysis.

### Step 2: Analyze Profiling Results

4. **Compute aggregate statistics** from captured attention weights.
   - Per-block, per-head entropy and sparsity.
   - Quadrant decomposition of attention energy.
   - Spatial locality correlation.

5. **Document findings** with visualizations in the profiling notebook.
   - Identify candidate blocks and strategies for masking.

### Step 3: Implement Masking Infrastructure (Phase 2)

6. **Create `AttentionMaskConfig` dataclass** to parameterize mask strategies.

7. **Implement `_build_attention_mask()`** method on `Flux2Transformer2DModel`.
   - Support all strategies (A–D) via config.
   - Return properly shaped mask tensors compatible with `dispatch_attention_fn`.

8. **Modify the `forward()` method** to construct and pass masks per-block.
   - The `attention_mask` parameter already flows through the processors — just need to construct the right mask.

### Step 4: Run Masking Experiments

9. **Create experiment notebook** (`2210/attention_masking_experiments.ipynb`).
   - Implement the priority experiment matrix (P0–P6).
   - Generate comparison grids and metric tables.

10. **Evaluate quality and efficiency** using the defined metrics and acceptance criteria.

### Step 5: Report Results

11. **Update this document** with empirical results, conclusions, and recommended configurations.

---

## 5. Technical Notes

### 5.1 Memory Considerations

Full attention weight tensors are large: for sequence length S = 4096 (e.g., 64×64 image tokens + text tokens), each layer's attention weights are `[1, 24, 4096, 4096] = 1 × 24 × 4096 × 4096 × 4 bytes ≈ 1.61 GB` in float32 per block. Mitigation strategies:

- Store attention weights in float16 or bfloat16.
- Only capture one head at a time, or average across heads.
- Subsample: capture every Nth token's attention row.
- Use streaming: compute metrics on-the-fly without storing full matrices.

### 5.2 Compatibility with Flash Attention

Flash Attention (and `F.scaled_dot_product_attention` with memory-efficient backends) does **not** materialize the full attention matrix. Our profiling processors compute weights explicitly for analysis, but the actual inference path remains unchanged. For masking:

- `F.scaled_dot_product_attention` supports the `attn_mask` parameter natively.
- Flash Attention v2 supports block-sparse masks via `flash_attn_func` with `block_sparse_mask`.
- For strategies requiring fine-grained sparse masks, we may need to fall back to the native (non-flash) attention backend during masked inference.

### 5.3 Existing Infrastructure

The pipeline at `src/diffusers/pipelines/flux2/pipeline_flux2_klein.py` already includes:
- `torch.profiler` integration with `_get_profiler()`, `_record()`, `_report_profiler_stats()`
- `enable_profiler` flag and `profile_event_names` tracking
- TensorBoard trace export

Our attention profiling builds on top of this infrastructure and adds attention-weight-specific capture.

### 5.4 File Locations

| Component | File |
|---|---|
| Transformer blocks & attention | `src/diffusers/models/transformers/transformer_flux2.py` |
| Attention dispatch | `src/diffusers/models/attention_dispatch.py` |
| Klein pipeline (profiling support) | `src/diffusers/pipelines/flux2/pipeline_flux2_klein.py` |
| Conversion script (config reference) | `scripts/convert_flux2_to_diffusers.py` |
| Playground notebook | `2210/Flux_2_playground.ipynb` |

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Attention weights are too large to store | Profiling fails or OOM | Use streaming metrics, subsample, or reduce resolution |
| Sparse masks are incompatible with Flash Attention | No speedup from masking | Fall back to native SDPA; investigate block-sparse Flash Attention |
| Attention patterns are prompt-dependent | No universal mask strategy | Profile across diverse prompts; use adaptive strategies |
| Quality degrades sharply with any masking | No viable mask configuration | Use very conservative masks; combine with fine-tuning |
| Attention patterns are uniform (no exploitable structure) | Masking not useful | Pivot to other optimization techniques (e.g., token merging, caching) |
