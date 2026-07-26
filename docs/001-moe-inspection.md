# 001-moe-inspection: Real MoE routing inspection for Unlimited-OCR

Notebook: `notebooks/000-unlimited-ocr-demo/001-moe-inspection.py`

This notebook inspects the real Mixture-of-Experts (MoE) routing behavior of `baidu/Unlimited-OCR` using the actual model weights and a forward pass. It does not use hand-drawn diagrams: every plot is generated from the model's hidden states, gate weights, and routing outputs.

## Purpose

- Load the Unlimited-OCR model from Hugging Face.
- Verify the architecture: 12 layers, dense layer 0, MoE layers 1–11, 64 experts, top-6 routing.
- Capture `MoEGate` routing decisions (`topk_idx`, `topk_weight`) for all MoE layers.
- Save and visualize gate scores, top-k selections, per-layer expert load, and aggregate expert usage.
- Persist all intermediate tensors, tables, and summaries for reproducible inspection.

## Prerequisites

- Repository root: `/workspace/learn-unlimited-ocr`.
- Python environment: `.venv` with the notebook dependencies installed.
- A CUDA-capable GPU (the notebook asserts `torch.cuda.is_available()`).
- Hugging Face access. Add `HF_TOKEN` to `.env` if needed for gated downloads; `baidu/Unlimited-OCR` is public.
- `transformers`, `torch`, `seaborn`, `matplotlib`, `pandas`, `numpy`, `IPython` (and optionally `jupytext` for `.py` notebook execution).

## How to run

Activate the environment and run the notebook in a Jupyter frontend:

```bash
cd /workspace/learn-unlimited-ocr
source .venv/bin/activate

# Option A: open in JupyterLab / VS Code and run all cells
jupyter lab

# Option B: convert the percent-script notebook to .ipynb and execute it
jupytext --to notebook notebooks/000-unlimited-ocr-demo/001-moe-inspection.py
jupyter nbconvert --to notebook --execute notebooks/000-unlimited-ocr-demo/001-moe-inspection.ipynb
```

The notebook is a `.py` percent-script file. Running it directly with `python` is not recommended because it uses `IPython.display.Markdown` inline rendering.

## High-level notebook structure

1. **Check Environment** — assert CUDA and select `bfloat16` if supported, otherwise `float16`.
2. **Load Model** — download and load `baidu/Unlimited-OCR` with `trust_remote_code=True` and `use_safetensors=True`.
3. **Architecture Summary** — identify dense layer 0 (`DeepseekV2MLP`) and MoE layers 1–11 (`DeepseekV2MoE`), print gate weight shapes.
4. **Capture MoE Routing** — register forward hooks on every `MoEGate` and run a short forward pass.
5. **Showcase: Input Tokens** — save tokenized inputs (ids and strings).
6. **Showcase: Top-K Routing Sample** — save routing tensor shapes and a per-token routing table for the target layer.
7. **Routing Table** — print a human-readable table of selected experts and weights per token.
8. **Gate Score Distribution** — capture the hidden state entering the target gate, compute the full 64-expert softmax, and plot a heatmap.
9. **Showcase: Hidden State and Gate Score Matrix** — save hidden-state stats/heatmap, the full gate-score CSV, and a top-5 score sample.
10. **Top-K Selection Matrix** — build a binary `[tokens, experts]` selection matrix across all MoE layers and plot it.
11. **Expert Load per Layer** — compute the total routing weight per expert per layer and plot a heatmap.
12. **Aggregate Expert Usage** — sum routing weights across all layers and plot a bar chart.
13. **Final Validation** — assert all expected PNGs and intermediate artifacts exist.
14. **Cheat Sheet** — print a concise summary of the MoE architecture and routing.

## Inputs

Default `CONFIG` at the top of the notebook:

| Parameter | Value |
|-----------|-------|
| `model_name` | `baidu/Unlimited-OCR` |
| `sample_text` | `"The quick brown fox jumps over the lazy dog."` |
| `target_moe_layer` | `1` |
| `num_experts` | `64` |
| `top_k` | `6` |
| `dtype` | `bfloat16` if supported, else `float16` |
| `out_dir` | `outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/001-moe-inspection` |

## Final outputs

Four 300 DPI Seaborn heatmaps/barplots are saved directly in `out_dir`:

| File | Description |
|------|-------------|
| `01_gate_scores.png` | Softmax gate score distribution over all 64 experts for each token in the target layer. |
| `02_topk_selections.png` | Binary top-k expert selection matrix across all MoE layers. |
| `03_expert_load.png` | Total routing weight per expert for each MoE layer. |
| `04_aggregate_usage.png` | Aggregate routing weight per expert across all MoE layers. |

## Intermediate outputs

All intermediate artifacts are saved under `out_dir/intermediate/`:

| File | Description |
|------|-------------|
| `00_input_tokens.json` | Token ids and token strings for the sample text. |
| `01_routing_shapes_all_layers.json` | `topk_idx`/`topk_weight` shapes and `aux_loss` for every MoE layer. |
| `01_routing_table_layer1.csv` | Per-token selected experts and weights for the target layer. |
| `02_hidden_state_summary.json` | Hidden-state shape, dtype, mean, std, min, max. |
| `02_hidden_state_heatmap.png` | Heatmap of the first 64 hidden dimensions entering the target gate. |
| `02_gate_scores.csv` | Full softmax score matrix `[seq_len, 64]` for the target layer. |
| `02_gate_scores_top5.csv` | Top-5 experts and scores per token position. |
| `03_topk_selection_matrix.csv` | Binary `[seq_len, 64]` top-k selection matrix. |
| `04_expert_load.csv` | Per-layer routing-weight load `[num_moe_layers, 64]`. |
| `05_aggregate_usage.csv` | Aggregate routing weight per expert. |

A `run.log` file is also written to `out_dir`.

## MoE routing in this model

Unlimited-OCR uses a DeepSeek-V2-style MoE decoder. The notebook confirms the following facts from the loaded weights:

- **Dense layer:** `layers[0].mlp` is a `DeepseekV2MLP` (one shared FFN used by every token).
- **MoE layers:** `layers[1]` through `layers[11]` are `DeepseekV2MoE` blocks.
- **Experts:** each MoE layer has 64 routed expert FFNs.
- **Top-k routing:** each token is routed to 6 experts (`num_experts_per_tok = 6`).
- **Gate module:** `layer.mlp.gate` is a `MoEGate` module. Its forward pass returns `(topk_idx, topk_weight, aux_loss)`:
  - `topk_idx`: indices of the selected experts per token, shape `[seq_len, 6]`.
  - `topk_weight`: routing weights for the selected experts, shape `[seq_len, 6]`.
  - `aux_loss`: optional auxiliary load-balancing loss.
- **Gate score distribution:** the full score over all 64 experts is computed by applying softmax to the linear projection of the hidden state through the gate weight matrix, `F.linear(hidden, gate.weight)` then `F.softmax(..., dim=-1)`.
