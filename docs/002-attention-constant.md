# 002-attention-constant — Verify Unlimited-OCR decoder memory stays constant

Companion to `notebooks/000-unlimited-ocr-demo/notebooks-py/002-attention-constant.py`.

## Purpose

Demonstrate that the Unlimited-OCR decoder's GPU memory stays flat during long
autoregressive generation. The model uses **R-SWA (Reference Sliding Window
Attention)**: every reference token (visual + prompt) is kept, and only the
latest `n` output tokens are kept in the KV cache. That bounds the cache at
`L_m + n` instead of the full-sequence `L_m + T` seen in vanilla attention.

## Prerequisites

- Model: `baidu/Unlimited-OCR` (public on Hugging Face).
- Hugging Face token in `.env`:

  ```bash
  HF_TOKEN=hf_...
  ```

- Python 3.12 venv at `.venv`.
- CUDA GPU with enough VRAM for the model at `bfloat16` or `float16`.

## Environment setup

```bash
cd /workspace/learn-unlimited-ocr
uv venv --python 3.12
source .venv/bin/activate
uv sync
export CUDA_VISIBLE_DEVICES=0
export $(grep -v '^#' .env | xargs)
```

## Run the notebook

The notebook is a `# %%` Python script. Run it end-to-end with the activated
venv:

```bash
python notebooks/000-unlimited-ocr-demo/notebooks-py/002-attention-constant.py
```

Or open it as a notebook in VS Code / Jupyter with Jupytext.

## High-level notebook structure

1. **Environment check** — pick `bfloat16` if the GPU supports it.
2. **Load model** — load `baidu/Unlimited-OCR` tokenizer and model.
3. **Inspect R-SWA config** — read `sliding_window`, attention class, and layer
   shapes.
4. **Load document pages** — download the Unlimited-OCR paper PDF, rasterize the
   first `max_pages` pages.
5. **Build multi-page inputs** — tokenize pages, insert image tokens, and build
   the model input tensors.
6. **Compute token budget** — compute `L_m`, `n`, and the upper-bound KV-cache
   size.
7. **Visualize receptive field** — draw the R-SWA attention mask.
8. **Profile memory during generation** — attach a forward hook and run
   generation while recording per-step GPU memory.
9. **Save trace and decode output** — write JSON trace, generated text, and a
   run summary.
10. **Plot measured memory and theory** — save the final matplotlib figures.
11. **Validate constant memory** — assert that allocated memory stops growing
    after the recent window is warm.
12. **Validate artifacts** — confirm every intermediate artifact exists.
13. **Summary and inline plots** — print results and display figures.

## Inputs

- **PDF**: `https://github.com/baidu/Unlimited-OCR/raw/main/Unlimited-OCR.pdf`
  (cached locally at `inputs/Unlimited-OCR.pdf`).
- **Pages**: first `max_pages` rasterized pages (default `2`) at `300` DPI.
- **Prompt**: `<image>Multi page parsing.`
- **Model**: `baidu/Unlimited-OCR`
- **Generation parameters**:
  - `max_length`: 2048
  - `no_repeat_ngram_size`: 5
  - `image_size`: 1024
  - `patch_size`: 16
  - `downsample_ratio`: 4
  - `sample_dpi`: 300
  - `max_pages`: 2

## Outputs

All artifacts are written under:

```text
outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/002-attention-constant/
```

### Final plots (`plots/`)

| Plot | File | What it shows |
| --- | --- | --- |
| Measured decoder memory | `decoder_memory_allocated_peak.png` | Allocated and peak GPU memory per forward step; prefill spike, then flat decode. |
| Theoretical KV-cache size | `kv_cache_theory.png` | `L_m + T` vs `L_m + min(T, n)`; R-SWA flattens at `L_m + n`. |
| R-SWA receptive field | `attention_receptive_field.png` | Mask: reference prefix always visible, only the latest `n` output tokens visible. |

### Results (`results/`)

- `decoder_memory_trace.json` — per-step allocated/peak MB and sequence length.
- `generated_text.txt` — raw decoded output (with special tokens).
- `run_summary.json` — reference tokens, recent window, generated length, upper-bound cache size, etc.

### Intermediate outputs (`intermediate/`)

- `00_input_page.png` — first rasterized input page.
- `01_token_budget.json` — visual/prompt/reference token counts, recent window, and upper-bound cache MB.
- `02_receptive_field_mask.npy` — raw R-SWA attention mask (`0=forgotten`, `1=reference`, `2=recent`).
- `02_receptive_field_mask_heatmap.png` — Seaborn heatmap of the mask.
- `03_memory_trace_sample.json` — first and last steps of the memory trace.
- `03_memory_trace_seaborn.png` — Seaborn line plot of allocated/peak memory.
- `04_generated_text_snippet.txt` — first 1000 characters of generated text.

### Other artifacts

- `run.log` — full notebook log.
- `inputs/Unlimited-OCR.pdf` — cached PDF.
- `inputs/pages/page_0001.png` (and `page_0002.png`) — rasterized pages.

## Key paper insight

Unlimited-OCR's R-SWA treats the visual tokens and the prompt as a fixed
reference prefix of length `L_m`. During decoding it keeps only the latest `n`
output tokens in the KV cache. Therefore:

```text
R-SWA KV cache  ≤ L_m + n
```

Vanilla full attention would store `L_m + T`, where `T` is the total generated
length. Because `n` is fixed, the decoder memory stops growing once the recent
window is full, which is exactly what the memory trace shows.
