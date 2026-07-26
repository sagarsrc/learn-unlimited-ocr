# learn-unlimited-ocr

Notebook experiments with `baidu/Unlimited-OCR`.

## Setup

Requires Python 3.12 and CUDA 12.x capable GPU.

```bash
uv sync
```

## Run

```bash
source .venv/bin/activate
python notebooks/000-unlimited-ocr-demo/000-run-unlimited-ocr.py
```

The notebook is configurable via `CONFIG` at the top:
- `source`: `"pdf"` or `"synthetic"`
- `pdf_path`: path to a PDF when `source="pdf"`
- `runs`: list of `RunConfig` entries to execute multiple modes/page counts in one pass

## Outputs

Generated artifacts are written to `outputs/` and ignored by git.
