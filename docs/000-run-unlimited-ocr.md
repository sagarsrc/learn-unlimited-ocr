# Run the `000-run-unlimited-ocr` notebook

Notebook: `notebooks/000-unlimited-ocr-demo/notebooks-py/000-run-unlimited-ocr.py`

This notebook loads `baidu/Unlimited-OCR` through `transformers`, renders a test document, and runs local single-page / multi-page inference. It also replicates the BASE-mode pipeline step-by-step so every intermediate artifact can be inspected.

## Purpose

- Demonstrate end-to-end document parsing with `baidu/Unlimited-OCR` without starting an SGLang server.
- Show three inference modes:
  - `single_gundam` — cropped patches (`crop_mode=True`, `image_size=640`).
  - `single_base` — single square view (`crop_mode=False`, `image_size=1024`).
  - `multi_page` — stitched multi-page context.
- Expose and save every intermediate artifact: preprocessed image, tokenized prompt, raw model string, parsed layout CSV, label distribution, and boxed result.

Default run is intentionally minimal: one page, BASE-only.

## Prerequisites

- Python 3.12+ and `uv`.
- Hugging Face token in `.env` if needed for model download or gated access:

```bash
echo 'HF_TOKEN=hf_...' > .env
source .env
```

- Virtual environment at `.venv` with project dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

- CUDA GPU. The notebook selects `bfloat16` when supported, otherwise `float16`.

## How to run

Run from the repo root so relative paths resolve:

```bash
cd /workspace/learn-unlimited-ocr
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python notebooks/000-unlimited-ocr-demo/notebooks-py/000-run-unlimited-ocr.py
```

The `.py` file uses `# %%` cells and is runnable as a script.

## Notebook structure

1. **Check Environment** — assert CUDA, pick dtype.
2. **Load Model** — load tokenizer and `baidu/Unlimited-OCR` with `trust_remote_code=True`.
3. **Load Document** — download / cache the PDF or render synthetic pages.
4. **Visualize Input** — display the first loaded page.
5. **Showcase: Input Image** — save `00_input_page.png` under `intermediate/`.
6. **Inference Runners** — execute all `RunConfig` entries (GUNDAM, BASE, multi-page).
7. **Showcase: BASE Pipeline Intermediates** — replicate BASE-mode preprocessing, tokenization, generation, parsing, and boxing.
8. **Inspect Outputs** — list every file produced by the run.
9. **Final Validation** — assert expected artifacts exist.
10. **Display Results** — render `result.md` and boxed images inline in a notebook frontend.

## Inputs

- **Document source**:
  - PDF (default): `https://github.com/baidu/Unlimited-OCR/raw/main/Unlimited-OCR.pdf`
  - Synthetic: set `CONFIG.source = "synthetic"` to render sample pages.
- **Cached PDF path**: `inputs/Unlimited-OCR.pdf` (downloaded once).
- **Page selection**: controlled by `RunConfig.max_pages`. Default `first_1_base` uses `max_pages=1`.
- **Preprocessing / inference parameters** (from `CONFIG`):

```python
base_size=1024              # square canvas used for padding
gundam_image_size=640       # patch size for GUNDAM (cropped) mode
base_image_size=1024        # patch size for BASE mode
multi_image_size=1024       # patch size for multi-page mode
max_length=32768
no_repeat_ngram_size=35
gundam_ngram_window=128
base_ngram_window=128
multi_ngram_window=1024
sample_dpi=300              # PDF rasterization DPI
```

- **Default run**: one page, BASE-only (`run_gundam=False`, `run_base=True`, `run_multi=False`).

## Outputs

Final outputs are written under:

```text
outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/000-run-unlimited-ocr/<run_name>/<mode>/
```

For the default run:

```text
outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/000-run-unlimited-ocr/first_1_base/single_base/
├── result.md                 # parsed markdown
└── result_with_boxes*.jpg    # page with layout boxes drawn
```

Per-mode directories (`single_gundam`, `single_base`, `multi_page`) are created only when the corresponding flag is enabled in `RunConfig`.

## Intermediate outputs

The BASE pipeline showcase saves the following under:

```text
outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/000-run-unlimited-ocr/intermediate/
```

| File | Description |
|------|-------------|
| `00_input_page.png` | Raw input page raster. |
| `01_preprocessed_global_view.png` | Padded square RGB view fed to the model. |
| `01_preprocessed_tensor.png` | Normalized `[3, H, W]` tensor saved as image. |
| `02_tokenized_prompt.json` | Token counts, image token ID, and `input_ids`. |
| `02_formatted_prompt.txt` | Text returned by `uocr.format_messages(...)`. |
| `03_raw_generated_text.txt` | Full decoded model output including `<|det|>` tokens. |
| `04_layout_predictions.csv` | Parsed `(label, x1, y1, x2, y2)` rows. |
| `04_label_distribution.png` | `seaborn` count plot of predicted labels. |
| `05_result_with_boxes.jpg` | Input page with parsed boxes overlaid. |

A run log is also written next to the outputs:

```text
outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/000-run-unlimited-ocr/run.log
```

## Expected artifact paths

Default run (`first_1_base`, BASE mode):

```text
outputs/<YYYY-MM-DD>/000-unlimited-ocr-demo/000-run-unlimited-ocr/
├── run.log
├── first_1_base/
│   └── single_base/
│       ├── result.md
│       └── result_with_boxes*.jpg
└── intermediate/
    ├── 00_input_page.png
    ├── 01_preprocessed_global_view.png
    ├── 01_preprocessed_tensor.png
    ├── 02_tokenized_prompt.json
    ├── 02_formatted_prompt.txt
    ├── 03_raw_generated_text.txt
    ├── 04_layout_predictions.csv
    ├── 04_label_distribution.png
    └── 05_result_with_boxes.jpg
```

`<YYYY-MM-DD>` is the date the notebook runs. The final validation cell asserts all of the above files exist and are non-empty.
