# How I made the Unlimited-OCR repo run

- **Repo**: `/workspace/Unlimited-OCR`
- **Machine**: Vast.ai container running **Ubuntu 24.04.4 LTS**
- **GPU**: **NVIDIA GeForce RTX 3090**, 24 GB VRAM
  - Compute capability: `8.6`
  - Driver version: `560.35.03`
  - CUDA Version reported by `nvidia-smi`: `12.6`
- **Tooling**: `uv 0.11.19`, Python `3.12.13`
- **Route**: SGLang via the bundled wheel and `infer.py`

## 1. Environment setup with `uv`

```bash
cd /workspace/Unlimited-OCR

# create a 3.12 venv
uv venv --python 3.12
source .venv/bin/activate

# install the bundled SGLang wheel
uv pip install wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl

# pin the kernels version and add PyMuPDF / requests
uv pip install kernels==0.11.7 pymupdf==1.27.2.2 requests
```

Key installed versions after the above:

```text
python        : 3.12.13 (conda-forge build)
uv            : 0.11.19
torch         : 2.9.1+cu128
transformers  : 5.3.0
sglang        : 0.0.0.dev11416+g92e8bb79e
flashinfer    : 0.6.7.post3
pymupdf       : 1.27.2.2
Pillow        : 12.3.0
requests      : 2.34.2
numpy         : 2.5.1
```

No Hugging Face token is required — `baidu/Unlimited-OCR` is public.

## 2. Run inference

The provided script `infer.py` starts its own SGLang server, waits for `/health`,
then sends the OpenAI-compatible chat requests.

```bash
cd /workspace/Unlimited-OCR
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python infer.py \
  --model_dir baidu/Unlimited-OCR \
  --image_dir assets \
  --output_dir outputs \
  --concurrency 1 \
  --gpu 0 \
  --image_mode gundam
```

`infer.py` uses the following SGLang server configuration (equivalent to the
command it spawns internally):

```text
python -m sglang.launch_server \
  --model baidu/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --attention-backend fa3 \
  --page-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port 10000
```

Client-side defaults used for the requests:

```text
prompt             : document parsing.
temperature        : 0
image_mode         : gundam  (base_size=1024, image_size=640, crop_mode=True)
max_length         : 32768
no_repeat_ngram_size: 35
ngram_window       : 128
max_retries        : 5
```

## 3. Observed result

```text
Starting SGLang server on GPU 0, port 10000 ...
Server PID: 25061
Server ready (176s)
Mode: dataset_images, requests=2, concurrency=1, image_mode=gundam
  [1] Unlimited-OCR.png: 31 tokens, 13.8s
  [2] baidu.png: 20 tokens, 0.1s

============================================================
Concurrent Results:
  Requests: 2/2
  Total tokens: 51
  Wall time: 22.91s
  System TPS: 2.23 tokens/s
  Avg tokens/request: 26
  Avg decode_time/request: 6.96s
============================================================
```

Both images under `assets/` were parsed successfully. Markdown outputs were written to `outputs/`.
