# %%
import datetime
import logging
import math
import os
import sys
import tempfile
import requests
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence, Tuple

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import Markdown, Image as IPImage, display
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Showcase:
    """Save intermediate artifacts, log summaries, and display them inline.

    Parameters
    ----------
    out_dir : Path
        Required. Notebook output directory; artifacts are written to
        ``out_dir / "intermediate"``.
    logger : logging.Logger
        Required. Logger used for artifact summaries.
    """

    def __init__(self, out_dir: Path, logger: logging.Logger):
        """Initialize the showcase helper.

        Parameters
        ----------
        out_dir : Path
            Required. Notebook output directory.
        logger : logging.Logger
            Required. Logger used for artifact summaries.
        """
        self.logger = logger
        self.dir = Path(out_dir) / "intermediate"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _save(self, name: str, writer: Callable[[Path], None]) -> Path:
        """Write an artifact and assert it is non-empty.

        Parameters
        ----------
        name : str
            Required. File name inside the intermediate directory.
        writer : Callable[[Path], None]
            Required. Function that writes the artifact to the given path.

        Returns
        -------
        Path
            Path to the written artifact.
        """
        path = self.dir / name
        writer(path)
        assert path.exists() and path.stat().st_size > 0, f"Empty artifact: {path}"
        self.logger.info("[Showcase] Saved %s (%d bytes)", path, path.stat().st_size)
        return path

    def log_summary(self, name: str, obj: Any) -> None:
        """Log a shape/dtype/length summary for an intermediate object.

        Parameters
        ----------
        name : str
            Required. Human-readable label for the object.
        obj : Any
            Required. Object to summarize.
        """
        if torch.is_tensor(obj):
            self.logger.info(
                "[Showcase] %s: tensor shape=%s dtype=%s",
                name,
                tuple(obj.shape),
                obj.dtype,
            )
        elif isinstance(obj, np.ndarray):
            self.logger.info(
                "[Showcase] %s: ndarray shape=%s dtype=%s",
                name,
                obj.shape,
                obj.dtype,
            )
        elif isinstance(obj, Image.Image):
            self.logger.info(
                "[Showcase] %s: PIL image size=%s mode=%s",
                name,
                obj.size,
                obj.mode,
            )
        elif isinstance(obj, (list, tuple, str)):
            self.logger.info(
                "[Showcase] %s: %s len=%d", name, type(obj).__name__, len(obj)
            )
        else:
            self.logger.info("[Showcase] %s: %s", name, type(obj).__name__)

    def save_image(self, image: Image.Image, name: str, inline: bool = True) -> Path:
        """Save a PIL image, log its size, and display it inline.

        Parameters
        ----------
        image : Image.Image
            Required. Image to save.
        name : str
            Required. Artifact file name.
        inline : bool
            Optional. Whether to display inline. Defaults to True.

        Returns
        -------
        Path
            Path to the saved image.
        """
        self.log_summary(name, image)
        path = self._save(name, lambda p: image.save(p))
        if inline:
            display(IPImage(filename=str(path)))
        return path

    def save_tensor_image(
        self, tensor: torch.Tensor, name: str, inline: bool = True
    ) -> Path:
        """Save a normalized CHW float tensor in [-1, 1] as an image.

        Parameters
        ----------
        tensor : torch.Tensor
            Required. Image tensor of shape [3, H, W] normalized to [-1, 1].
        name : str
            Required. Artifact file name.
        inline : bool
            Optional. Whether to display inline. Defaults to True.

        Returns
        -------
        Path
            Path to the saved image.
        """
        self.log_summary(name, tensor)
        arr = tensor.detach().float().cpu()
        arr = ((arr + 1.0) / 2.0).clamp(0, 1)
        img = Image.fromarray((arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        path = self._save(name, lambda p: img.save(p))
        if inline:
            display(IPImage(filename=str(path)))
        return path

    def save_text(
        self,
        text: str,
        name: str,
        preview_chars: int = 1500,
        inline: bool = True,
    ) -> Path:
        """Save text, log its length, and display a fenced preview.

        Parameters
        ----------
        text : str
            Required. Text to save.
        name : str
            Required. Artifact file name.
        preview_chars : int
            Optional. Number of characters shown inline. Defaults to 1500.
        inline : bool
            Optional. Whether to display inline. Defaults to True.

        Returns
        -------
        Path
            Path to the saved text file.
        """
        self.log_summary(name, text)
        path = self._save(name, lambda p: p.write_text(text, encoding="utf-8"))
        if inline:
            preview = text[:preview_chars]
            if len(text) > preview_chars:
                preview += f"\n... [{len(text) - preview_chars} more chars]"
            display(Markdown(f"```text\n{preview}\n```"))
        return path

    def save_json(
        self, obj: Any, name: str, inline: bool = True, max_lines: int = 40
    ) -> Path:
        """Save an object as JSON and display a pretty preview.

        Parameters
        ----------
        obj : Any
            Required. JSON-serializable object.
        name : str
            Required. Artifact file name.
        inline : bool
            Optional. Whether to display inline. Defaults to True.
        max_lines : int
            Optional. Number of JSON lines shown inline. Defaults to 40.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        payload = json.dumps(obj, indent=2, default=str)
        path = self._save(name, lambda p: p.write_text(payload, encoding="utf-8"))
        self.logger.info("[Showcase] %s: %d bytes of JSON", name, len(payload))
        if inline:
            lines = payload.splitlines()
            preview = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                preview += f"\n... [{len(lines) - max_lines} more lines]"
            display(Markdown(f"```json\n{preview}\n```"))
        return path

    def save_table(self, df: pd.DataFrame, name: str, inline: bool = True) -> Path:
        """Save a dataframe as CSV, log its shape, and display it inline.

        Parameters
        ----------
        df : pd.DataFrame
            Required. Dataframe to save.
        name : str
            Required. Artifact file name.
        inline : bool
            Optional. Whether to display inline. Defaults to True.

        Returns
        -------
        Path
            Path to the saved CSV file.
        """
        self.logger.info(
            "[Showcase] %s: DataFrame shape=%s columns=%s",
            name,
            df.shape,
            list(df.columns),
        )
        path = self._save(name, lambda p: df.to_csv(p, index=False))
        if inline:
            display(df)
        return path

    def save_figure(
        self, fig: plt.Figure, name: str, dpi: int = 300, inline: bool = True
    ) -> Path:
        """Save a matplotlib figure, display it inline, and close it.

        Parameters
        ----------
        fig : plt.Figure
            Required. Figure to save.
        name : str
            Required. Artifact file name.
        dpi : int
            Optional. Save resolution. Defaults to 300.
        inline : bool
            Optional. Whether to display inline. Defaults to True.

        Returns
        -------
        Path
            Path to the saved figure.
        """
        path = self._save(name, lambda p: fig.savefig(p, dpi=dpi, bbox_inches="tight"))
        if inline:
            display(IPImage(filename=str(path)))
        plt.close(fig)
        return path


@dataclass(frozen=True)
class RunConfig:
    """One inference run to execute.

    Parameters
    ----------
    name : str
        Required. Subdirectory name for this run.
    max_pages : int
        Required. Pages to use from the loaded document.
    run_gundam : bool
        Required. Whether to run GUNDAM single-page inference.
    run_base : bool
        Required. Whether to run BASE single-page inference.
    run_multi : bool
        Required. Whether to run multi-page inference.
    """

    name: str
    max_pages: int
    run_gundam: bool
    run_base: bool
    run_multi: bool


@dataclass
class Config:
    """Paths and hyperparameters for the Unlimited-OCR demo.

    Parameters
    ----------
    work_dir : Path
        Required. Project root used to resolve relative paths.
    out_dir : Path
        Required. Directory for all generated artifacts.
    model_name : str
        Required. Hugging Face model identifier.
    source : str
        Required. Document source, either "pdf" or "synthetic".
    pdf_url : str | None
        Optional. URL to download the PDF when source is "pdf".
    pdf_cache_path : Path
        Required. Local path to cache the downloaded PDF.
    synthetic_page_count : int
        Required. Number of synthetic pages to render when source is "synthetic".
    runs : Sequence[RunConfig]
        Required. Inference runs to execute.
    base_size : int
        Required. Base canvas size for image preprocessing.
    gundam_image_size : int
        Required. Patch size for GUNDAM (cropped) mode.
    base_image_size : int
        Required. Patch size for BASE (single-view) mode.
    multi_image_size : int
        Required. Patch size for multi-page mode.
    max_length : int
        Required. Maximum generation length.
    no_repeat_ngram_size : int
        Required. N-gram repetition suppression width.
    gundam_ngram_window : int
        Required. Local window for GUNDAM repetition check.
    base_ngram_window : int
        Required. Local window for BASE repetition check.
    multi_ngram_window : int
        Required. Local window for multi-page repetition check.
    sample_dpi : int
        Required. DPI used when rasterizing the PDF.
    font_candidates : Sequence[str]
        Optional. Paths to try for the sample font.
    dtype : torch.dtype | None
        Optional. Selected torch dtype, set at runtime.
    """

    work_dir: Path
    out_dir: Path
    model_name: str
    source: str
    pdf_url: str | None
    pdf_cache_path: Path
    synthetic_page_count: int
    runs: Sequence[RunConfig]
    base_size: int
    gundam_image_size: int
    base_image_size: int
    multi_image_size: int
    max_length: int
    no_repeat_ngram_size: int
    gundam_ngram_window: int
    base_ngram_window: int
    multi_ngram_window: int
    sample_dpi: int
    font_candidates: Sequence[str] = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    dtype: torch.dtype | None = None

    def replace(self, **kwargs: object) -> "Config":
        """Return a new Config with the given fields replaced."""
        return replace(self, **kwargs)


TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
CONFIG = Config(
    work_dir=Path.cwd(),
    out_dir=Path(f"outputs/{TODAY}/000-unlimited-ocr-demo/000-run-unlimited-ocr"),
    model_name="baidu/Unlimited-OCR",
    source="pdf",
    pdf_url="https://github.com/baidu/Unlimited-OCR/raw/main/Unlimited-OCR.pdf",
    pdf_cache_path=Path("inputs/Unlimited-OCR.pdf"),
    synthetic_page_count=3,
    # Minimal committed default: one page, BASE mode only (fast dev loop).
    # Re-enable GUNDAM / multi-page or add more pages when needed, e.g.
    # RunConfig(name="first_4_all", max_pages=4, run_gundam=True,
    #           run_base=True, run_multi=True).
    runs=(
        RunConfig(
            name="first_1_base",
            max_pages=1,
            run_gundam=False,
            run_base=True,
            run_multi=False,
        ),
    ),
    base_size=1024,
    gundam_image_size=640,
    base_image_size=1024,
    multi_image_size=1024,
    max_length=32768,
    no_repeat_ngram_size=35,
    gundam_ngram_window=128,
    base_ngram_window=128,
    multi_ngram_window=1024,
    sample_dpi=300,
)

# Persist logs alongside outputs for this run.
CONFIG.out_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(CONFIG.out_dir / "run.log", mode="w")
_file_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_file_handler)

logger.info(
    "Config: source=%s pdf_url=%s runs=%s",
    CONFIG.source,
    CONFIG.pdf_url,
    [r.name for r in CONFIG.runs],
)
if CONFIG.source == "pdf":
    assert CONFIG.pdf_url, "pdf_url required when source=pdf"


# %% [markdown]
# ## Check Environment
# Verify CUDA availability and pick the best dtype for the current GPU.


# %%
class EnvironmentChecker:
    """Check GPU availability and select the inference dtype.

    Parameters
    ----------
    config : Config
        Required. Global configuration; updated with the selected dtype.
    """

    def __init__(self, config: Config):
        """Initialize with global config.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        """
        self.config = config

    def check(self) -> torch.dtype:
        """Assert CUDA is available and return the selected dtype.

        Returns
        -------
        torch.dtype
            bfloat16 if supported, otherwise float16.
        """
        assert torch.cuda.is_available(), "CUDA unavailable"
        gpu_name = torch.cuda.get_device_name(0)
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        logger.info("[EnvironmentChecker] GPU: %s", gpu_name)
        logger.info("[EnvironmentChecker] dtype: %s", dtype)
        return dtype


env_checker = EnvironmentChecker(CONFIG)
CONFIG = CONFIG.replace(dtype=env_checker.check())
logger.info("Outputs: CONFIG.dtype=%s", CONFIG.dtype)


# %% [markdown]
# ## Load Model
# Load the Unlimited-OCR tokenizer and model onto the GPU.


# %%
class ModelLoader:
    """Load the Unlimited-OCR model and tokenizer.

    Parameters
    ----------
    config : Config
        Required. Global configuration with model_name and dtype.
    """

    def __init__(self, config: Config):
        """Initialize with global config.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        """
        self.config = config

    def load(self) -> Tuple[AutoTokenizer, AutoModel]:
        """Download and load the model.

        Returns
        -------
        Tuple[AutoTokenizer, AutoModel]
            Tokenizer and model ready for inference.
        """
        logger.info("[ModelLoader] Loading %s ...", self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=self.config.dtype,
        )
        model = model.eval().cuda()
        logger.info("[ModelLoader] Model loaded")
        return tokenizer, model


model_loader = ModelLoader(CONFIG)
tokenizer, model = model_loader.load()
logger.info(
    "Outputs: tokenizer=%s model=%s",
    type(tokenizer).__name__,
    type(model).__name__,
)


# %% [markdown]
# ## Load Document
# Rasterize the PDF or render synthetic pages for the experiment.


# %%
class PdfRasterizer:
    """Rasterize PDF pages to PNGs.

    Parameters
    ----------
    dpi : int
        Required. Rendering DPI.
    """

    def __init__(self, dpi: int):
        """Initialize with rendering DPI.

        Parameters
        ----------
        dpi : int
            Required. Rendering DPI.
        """
        self.dpi = dpi

    def rasterize(self, pdf_path: Path, max_pages: int) -> Tuple[Path, List[Path]]:
        """Convert a PDF to PNGs, keeping at most max_pages.

        Parameters
        ----------
        pdf_path : Path
            Required. PDF to rasterize.
        max_pages : int
            Required. Maximum number of pages to keep.

        Returns
        -------
        Tuple[Path, List[Path]]
            Temporary directory and paths to the generated PNGs in page order.
        """
        doc = fitz.open(pdf_path)
        tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_ocr_"))
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        paths: List[Path] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            out_path = tmp_dir / f"page_{i + 1:04d}.png"
            page.get_pixmap(matrix=mat).save(str(out_path))
            paths.append(out_path)
        doc.close()
        logger.info(
            "[PdfRasterizer] Rasterized %s pages to %s",
            len(paths),
            tmp_dir,
        )
        return tmp_dir, paths


class SamplePageRenderer:
    """Render synthetic document pages with headings, paragraphs, and tables.

    Parameters
    ----------
    config : Config
        Required. Global configuration; out_dir is used for inputs subdir.
    """

    def __init__(self, config: Config):
        """Initialize with global config.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        """
        self.config = config
        self.inputs_dir = config.work_dir / "inputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Return the first available font at the requested size.

        Parameters
        ----------
        size : int
            Required. Font size in points.

        Returns
        -------
        ImageFont.FreeTypeFont
            Loaded font.
        """
        for path in self.config.font_candidates:
            font_path = Path(path)
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size)
        return ImageFont.load_default()

    def render(self, page_no: int) -> Path:
        """Render one sample page.

        Parameters
        ----------
        page_no : int
            Required. Page number printed in the page title.

        Returns
        -------
        Path
            Path to the written PNG file.
        """
        w, h = 1240, 1754
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        title_font = self._load_font(48)
        head_font = self._load_font(34)
        body_font = self._load_font(26)

        draw.text(
            (80, 70),
            f"Quarterly Operations Report — Page {page_no}",
            fill="black",
            font=title_font,
        )
        draw.line([(80, 145), (w - 80, 145)], fill="black", width=3)

        body = (
            "This document demonstrates Unlimited-OCR's one-shot long-horizon "
            "parsing. The model reads an entire page — headings, paragraphs, "
            "and tables — and emits structured text in a single decoding pass. "
            "Unlike classic OCR pipelines, no separate layout-analysis stage "
            "is required."
        )
        y = 190
        for line in textwrap.wrap(body, width=72):
            draw.text((80, y), line, fill="black", font=body_font)
            y += 40

        y += 30
        draw.text(
            (80, y),
            f"Table {page_no}: Regional Revenue (USD, millions)",
            fill="black",
            font=head_font,
        )
        y += 60
        rows = [
            ["Region", "Q1", "Q2", "Q3"],
            ["North", "12.4", "13.1", "15.0"],
            ["South", "9.8", "10.2", "11.7"],
            ["East", "14.3", "13.9", "16.2"],
            ["West", "11.1", "12.5", "12.9"],
        ]
        col_w, row_h, x0 = 260, 56, 80
        for row in rows:
            for c, cell in enumerate(row):
                x = x0 + c * col_w
                draw.rectangle(
                    [x, y, x + col_w, y + row_h],
                    outline="black",
                    width=2,
                )
                draw.text((x + 14, y + 12), cell, fill="black", font=body_font)
            y += row_h

        y += 50
        footer = (
            f"Note {page_no}: Figures are illustrative. Multi-page mode stitches "
            "context across pages, so cross-page references remain coherent."
        )
        for line in textwrap.wrap(footer, width=72):
            draw.text((80, y), line, fill="black", font=body_font)
            y += 40

        path = self.inputs_dir / f"sample_page_{page_no}.png"
        img.save(path)
        return path

    def render_all(self, page_numbers: Sequence[int]) -> List[Path]:
        """Render a list of page numbers.

        Parameters
        ----------
        page_numbers : Sequence[int]
            Required. Page numbers to render.

        Returns
        -------
        List[Path]
            Paths to the rendered pages in order.
        """
        paths = [self.render(n) for n in page_numbers]
        logger.info(
            "[SamplePageRenderer] Wrote %s pages to %s",
            len(paths),
            self.inputs_dir,
        )
        return paths


class DocumentLoader:
    """Load pages for the experiment.

    Parameters
    ----------
    config : Config
        Required. Global configuration.
    """

    def __init__(self, config: Config):
        """Initialize with global config.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        """
        self.config = config

    def load(self) -> List[Path]:
        """Load document pages according to CONFIG.source.

        Downloads the PDF once to pdf_cache_path when source is "pdf".

        Returns
        -------
        List[Path]
            Image paths for the experiment.
        """
        max_pages = max(run.max_pages for run in self.config.runs)
        if self.config.source == "pdf":
            assert self.config.pdf_url is not None
            cache_path = self.config.pdf_cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                logger.info(
                    "[DocumentLoader] Downloading PDF from %s",
                    self.config.pdf_url,
                )
                response = requests.get(self.config.pdf_url, timeout=120)
                response.raise_for_status()
                cache_path.write_bytes(response.content)
                logger.info(
                    "[DocumentLoader] Saved PDF to %s (%s bytes)",
                    cache_path,
                    len(response.content),
                )
            rasterizer = PdfRasterizer(self.config.sample_dpi)
            _, paths = rasterizer.rasterize(cache_path, max_pages)
            return paths
        renderer = SamplePageRenderer(self.config)
        count = max(max_pages, self.config.synthetic_page_count)
        return renderer.render_all(range(1, count + 1))


document_loader = DocumentLoader(CONFIG)
page_images = document_loader.load()
logger.info(
    "Outputs: loaded %s pages, first=%s",
    len(page_images),
    page_images[0],
)
assert len(page_images) >= max(
    run.max_pages for run in CONFIG.runs
), "Not enough pages for configured runs"
for p in page_images:
    assert p.exists(), f"Missing page: {p}"


# %% [markdown]
# ## Visualize Input
# Preview the first loaded page.


# %%
class Visualizer:
    """Display an image inline with matplotlib.

    Parameters
    ----------
    image_path : Path
        Required. Path to the image to display.
    """

    def __init__(self, image_path: Path):
        """Initialize with image path.

        Parameters
        ----------
        image_path : Path
            Required. Path to the image to display.
        """
        self.image_path = image_path

    def show(self, title: str = "Input document") -> None:
        """Show the image.

        Parameters
        ----------
        title : str
            Optional. Title for the plot. Defaults to "Input document".
        """
        plt.figure(figsize=(6, 8))
        plt.imshow(Image.open(self.image_path))
        plt.axis("off")
        plt.title(title)
        plt.show()
        logger.info("[Visualizer] Displayed %s", self.image_path)


visualizer = Visualizer(page_images[0])
visualizer.show("First page of loaded document")


# %% [markdown]
# ## Showcase: Input Image
# Save the raw input page to the `intermediate/` folder, log its shape, and
# display it inline.


# %%
showcase = Showcase(CONFIG.out_dir, logger)


class InputImageShowcase:
    """Save and display the raw input page image.

    Parameters
    ----------
    showcase : Showcase
        Required. Shared save/log/display helper.
    """

    def __init__(self, showcase: Showcase):
        """Initialize with the shared showcase helper.

        Parameters
        ----------
        showcase : Showcase
            Required. Shared save/log/display helper.
        """
        self.showcase = showcase

    def run(self, image_path: Path) -> Image.Image:
        """Save and display the input page.

        Parameters
        ----------
        image_path : Path
            Required. Path to the input page image.

        Returns
        -------
        Image.Image
            Loaded RGB input image.
        """
        image = Image.open(image_path).convert("RGB")
        self.showcase.save_image(image, "00_input_page.png")
        logger.info("[InputImageShowcase] input size (W, H)=%s", image.size)
        return image


input_showcase = InputImageShowcase(showcase)
input_image = input_showcase.run(page_images[0])
logger.info("Outputs: input_image.size=%s", input_image.size)
assert input_image.size[0] > 0 and input_image.size[1] > 0


# %% [markdown]
# ## Inference Runners
# Run single-page and multi-page inference across the configured runs.


# %%
class SingleImageInferer:
    """Run Unlimited-OCR on a single image.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Required. Loaded tokenizer.
    model : AutoModel
        Required. Loaded model.
    """

    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel):
        """Initialize with tokenizer and model.

        Parameters
        ----------
        tokenizer : AutoTokenizer
            Required. Loaded tokenizer.
        model : AutoModel
            Required. Loaded model.
        """
        self.tokenizer = tokenizer
        self.model = model

    def infer(
        self,
        image_path: Path,
        output_dir: Path,
        prompt: str,
        image_size: int,
        crop_mode: bool,
        ngram_window: int,
    ) -> None:
        """Run inference and save results.

        Parameters
        ----------
        image_path : Path
            Required. Path to input image.
        output_dir : Path
            Required. Directory to write results.
        prompt : str
            Required. Prompt prefix.
        image_size : int
            Required. Patch size.
        crop_mode : bool
            Required. Whether to crop/overlap patches.
        ngram_window : int
            Required. Repetition window.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=CONFIG.base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            max_length=CONFIG.max_length,
            no_repeat_ngram_size=CONFIG.no_repeat_ngram_size,
            ngram_window=ngram_window,
            save_results=True,
        )
        logger.info("[SingleImageInferer] Wrote results to %s", output_dir)


class MultiPageInferer:
    """Run Unlimited-OCR over multiple images.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Required. Loaded tokenizer.
    model : AutoModel
        Required. Loaded model.
    """

    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel):
        """Initialize with tokenizer and model.

        Parameters
        ----------
        tokenizer : AutoTokenizer
            Required. Loaded tokenizer.
        model : AutoModel
            Required. Loaded model.
        """
        self.tokenizer = tokenizer
        self.model = model

    def infer(
        self,
        image_paths: Sequence[Path],
        output_dir: Path,
        prompt: str,
        image_size: int,
        ngram_window: int,
    ) -> None:
        """Run multi-page inference and save results.

        Parameters
        ----------
        image_paths : Sequence[Path]
            Required. Paths to images.
        output_dir : Path
            Required. Directory to write results.
        prompt : str
            Required. Prompt prefix.
        image_size : int
            Required. Patch size.
        ngram_window : int
            Required. Repetition window.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.infer_multi(
            self.tokenizer,
            prompt=prompt,
            image_files=[str(p) for p in image_paths],
            output_path=str(output_dir),
            image_size=image_size,
            max_length=CONFIG.max_length,
            no_repeat_ngram_size=CONFIG.no_repeat_ngram_size,
            ngram_window=ngram_window,
            save_results=True,
        )
        logger.info("[MultiPageInferer] Wrote results to %s", output_dir)


class RunOrchestrator:
    """Execute all configured inference runs.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Required. Loaded tokenizer.
    model : AutoModel
        Required. Loaded model.
    images : Sequence[Path]
        Required. Loaded document pages.
    config : Config
        Required. Global configuration.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        images: Sequence[Path],
        config: Config,
    ):
        """Initialize with tokenizer, model, images, and config."""
        self.tokenizer = tokenizer
        self.model = model
        self.images = images
        self.config = config

    def run(self) -> None:
        """Run every configured inference run."""
        single_inferer = SingleImageInferer(self.tokenizer, self.model)
        multi_inferer = MultiPageInferer(self.tokenizer, self.model)
        for run in self.config.runs:
            run_dir = self.config.out_dir / run.name
            run_dir.mkdir(parents=True, exist_ok=True)
            pages = list(self.images[: run.max_pages])
            logger.info(
                "[RunOrchestrator] Starting run '%s' with %s pages",
                run.name,
                len(pages),
            )
            if run.run_gundam:
                single_inferer.infer(
                    pages[0],
                    run_dir / "single_gundam",
                    prompt="<image>document parsing.",
                    image_size=self.config.gundam_image_size,
                    crop_mode=True,
                    ngram_window=self.config.gundam_ngram_window,
                )
            if run.run_base:
                single_inferer.infer(
                    pages[0],
                    run_dir / "single_base",
                    prompt="<image>document parsing.",
                    image_size=self.config.base_image_size,
                    crop_mode=False,
                    ngram_window=self.config.base_ngram_window,
                )
            if run.run_multi:
                multi_inferer.infer(
                    pages,
                    run_dir / "multi_page",
                    prompt="<image>Multi page parsing.",
                    image_size=self.config.multi_image_size,
                    ngram_window=self.config.multi_ngram_window,
                )
            logger.info(
                "[RunOrchestrator] Finished run '%s'",
                run.name,
            )


orchestrator = RunOrchestrator(tokenizer, model, page_images, CONFIG)
orchestrator.run()
logger.info("Outputs: all runs complete")


# %% [markdown]
# ## Showcase: BASE Pipeline Intermediates
# Replicate the BASE (non-crop) single-page pipeline step by step, using the
# model's own dynamically loaded helpers, so every intermediate can be saved,
# logged, and displayed: preprocessed image, tokenized prompt, raw generated
# string, parsed layout predictions, and the boxed result image.


# %%
def get_model_module(model: AutoModel):
    """Return the dynamically loaded remote-code modeling module.

    Parameters
    ----------
    model : AutoModel
        Required. Loaded model (its module is already in sys.modules).

    Returns
    -------
    module
        The loaded `modeling_unlimitedocr` module.
    """
    for module_name, module in sys.modules.items():
        if module_name.endswith("modeling_unlimitedocr"):
            return module
    raise RuntimeError("modeling_unlimitedocr module not found in sys.modules")


uocr = get_model_module(model)
logger.info("Outputs: uocr module=%s", uocr.__name__)


# %% [markdown]
# ### Showcase: Preprocessed Image
# BASE mode pads the page onto a square `base_image_size` canvas and normalizes
# it to a `[3, H, W]` tensor in `[-1, 1]`.


# %%
class BasePreprocessor:
    """Replicate BASE-mode (non-crop) preprocessing and showcase its outputs.

    Parameters
    ----------
    config : Config
        Required. Global configuration.
    showcase : Showcase
        Required. Shared save/log/display helper.
    """

    def __init__(self, config: Config, showcase: Showcase):
        """Initialize with config and showcase helper.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        showcase : Showcase
            Required. Shared save/log/display helper.
        """
        self.config = config
        self.showcase = showcase

    def run(self, image: Image.Image) -> torch.Tensor:
        """Pad and normalize the page, saving both views.

        Parameters
        ----------
        image : Image.Image
            Required. Raw input page image.

        Returns
        -------
        torch.Tensor
            Normalized image tensor of shape [3, image_size, image_size].
        """
        size = self.config.base_image_size
        transform = uocr.BasicImageTransform(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )
        pad_color = tuple(int(x * 255) for x in transform.mean)
        global_view = ImageOps.pad(image, (size, size), color=pad_color)
        self.showcase.save_image(global_view, "01_preprocessed_global_view.png")
        tensor = transform(global_view).to(self.config.dtype)
        self.showcase.save_tensor_image(tensor, "01_preprocessed_tensor.png")
        logger.info(
            "[BasePreprocessor] tensor shape=%s dtype=%s",
            tuple(tensor.shape),
            tensor.dtype,
        )
        return tensor


base_preprocessor = BasePreprocessor(CONFIG, showcase)
image_tensor = base_preprocessor.run(input_image)
logger.info("Outputs: image_tensor.shape=%s", tuple(image_tensor.shape))
assert image_tensor.shape == (3, CONFIG.base_image_size, CONFIG.base_image_size)


# %% [markdown]
# ### Showcase: Tokenized Prompt
# The formatted prompt is split around `<image>`; each image contributes
# `(num_queries + 1) * num_queries + 1` visual tokens.


# %%
class PromptTokenizer:
    """Replicate BASE-mode tokenization and showcase the tokenized prompt.

    Parameters
    ----------
    config : Config
        Required. Global configuration.
    tokenizer : AutoTokenizer
        Required. Loaded tokenizer.
    showcase : Showcase
        Required. Shared save/log/display helper.
    """

    IMAGE_TOKEN = "<image>"
    BOS_ID = 0

    def __init__(self, config: Config, tokenizer: AutoTokenizer, showcase: Showcase):
        """Initialize with config, tokenizer, and showcase helper.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        tokenizer : AutoTokenizer
            Required. Loaded tokenizer.
        showcase : Showcase
            Required. Shared save/log/display helper.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.showcase = showcase

    def run(self, prompt: str) -> dict:
        """Tokenize the prompt exactly as BASE mode does.

        Parameters
        ----------
        prompt : str
            Required. Prompt containing a single `<image>` token.

        Returns
        -------
        dict
            Mapping with input_ids, images_seq_mask, and counts.
        """
        formatted = uocr.format_messages(
            conversations=[
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ],
            sft_format="plain",
            system_prompt="",
        )
        text_splits = formatted.split(self.IMAGE_TOKEN)
        assert len(text_splits) == 2, "Prompt must contain exactly one <image> token"

        image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        if not isinstance(image_token_id, int) or image_token_id < 0:
            image_token_id = 128815  # fallback used by the reference model

        num_queries = math.ceil((self.config.base_image_size // 16) / 4)
        tokenized_image = (
            [image_token_id] * num_queries + [image_token_id]
        ) * num_queries + [image_token_id]
        before_ids = self.tokenizer.encode(text_splits[0], add_special_tokens=False)
        after_ids = self.tokenizer.encode(text_splits[1], add_special_tokens=False)

        input_ids = [self.BOS_ID] + before_ids + tokenized_image + after_ids
        images_seq_mask = (
            [False] * (1 + len(before_ids))
            + [True] * len(tokenized_image)
            + [False] * len(after_ids)
        )
        result = {
            "formatted_prompt": formatted,
            "image_token_id": image_token_id,
            "num_queries_per_side": num_queries,
            "num_tokens": len(input_ids),
            "num_visual_tokens": len(tokenized_image),
            "num_text_tokens": 1 + len(before_ids) + len(after_ids),
            "input_ids": input_ids,
            "images_seq_mask": images_seq_mask,
        }
        self.showcase.save_json(
            {k: v for k, v in result.items() if k != "images_seq_mask"},
            "02_tokenized_prompt.json",
        )
        self.showcase.save_text(formatted, "02_formatted_prompt.txt")
        logger.info(
            "[PromptTokenizer] num_tokens=%d visual=%d text=%d",
            result["num_tokens"],
            result["num_visual_tokens"],
            result["num_text_tokens"],
        )
        return result


prompt_tokenizer = PromptTokenizer(CONFIG, tokenizer, showcase)
tokenized = prompt_tokenizer.run("<image>document parsing.")
logger.info("Outputs: tokenized num_tokens=%d", tokenized["num_tokens"])
assert tokenized["num_visual_tokens"] > 0
assert len(tokenized["input_ids"]) == len(tokenized["images_seq_mask"])


# %% [markdown]
# ### Showcase: Raw Generated String
# Generate with the same settings as `model.infer` BASE mode and keep the raw
# decoded string, including `<|ref|>...` / `<|det|>...` special tokens.


# %%
class ShowcaseGenerator:
    """Generate exactly like BASE mode and showcase the raw model string.

    Parameters
    ----------
    config : Config
        Required. Global configuration.
    tokenizer : AutoTokenizer
        Required. Loaded tokenizer.
    model : AutoModel
        Required. Loaded model.
    showcase : Showcase
        Required. Shared save/log/display helper.
    """

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        showcase: Showcase,
    ):
        """Initialize with config, tokenizer, model, and showcase helper.

        Parameters
        ----------
        config : Config
            Required. Global configuration.
        tokenizer : AutoTokenizer
            Required. Loaded tokenizer.
        model : AutoModel
            Required. Loaded model.
        showcase : Showcase
            Required. Shared save/log/display helper.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.showcase = showcase

    def run(self, tokenized: dict, image_tensor: torch.Tensor) -> str:
        """Run generation and save the raw decoded string.

        Parameters
        ----------
        tokenized : dict
            Required. Output of PromptTokenizer.run.
        image_tensor : torch.Tensor
            Required. Preprocessed image tensor.

        Returns
        -------
        str
            Raw decoded model output including special tokens.
        """
        input_ids_t = torch.LongTensor(tokenized["input_ids"]).unsqueeze(0).cuda()
        seq_mask = (
            torch.tensor(tokenized["images_seq_mask"], dtype=torch.bool)
            .unsqueeze(0)
            .cuda()
        )
        images_ori = image_tensor.unsqueeze(0).cuda()
        images_crop = torch.zeros(
            (1, 3, self.config.base_size, self.config.base_size),
            dtype=self.config.dtype,
        ).cuda()
        spatial_crop = torch.tensor([[1, 1]], dtype=torch.long)

        orig_sw = getattr(self.model.config, "sliding_window_size", None) or getattr(
            self.model.config, "sliding_window", None
        )
        self.model.config._ring_window = orig_sw
        self.model.config.sliding_window = None
        try:
            with torch.autocast("cuda", dtype=self.config.dtype), torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids_t,
                    images=[(images_crop, images_ori)],
                    images_seq_mask=seq_mask,
                    images_spatial_crop=spatial_crop,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.config.max_length,
                    use_cache=True,
                    logits_processor=[
                        uocr.SlidingWindowNoRepeatNgramProcessor(
                            self.config.no_repeat_ngram_size,
                            self.config.base_ngram_window,
                        )
                    ],
                )
        finally:
            self.model.config.sliding_window = orig_sw

        generated_ids = output_ids[0, input_ids_t.shape[1] :]
        self.showcase.log_summary("generated_ids", generated_ids)
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        self.showcase.save_text(
            raw_text, "03_raw_generated_text.txt", preview_chars=2000
        )
        return raw_text


showcase_generator = ShowcaseGenerator(CONFIG, tokenizer, model, showcase)
raw_output = showcase_generator.run(tokenized, image_tensor)
logger.info("Outputs: raw_output len=%d", len(raw_output))
assert len(raw_output) > 0
assert "<|det|>" in raw_output, "Expected grounding dets in the raw output"


# %% [markdown]
# ### Showcase: Parsed Layout Predictions
# Parse the `<|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>` spans into a
# table of (label, box coordinates) and plot the label distribution.


# %%
class LayoutPredictionParser:
    """Parse grounding spans from the raw string into a prediction table.

    Parameters
    ----------
    showcase : Showcase
        Required. Shared save/log/display helper.
    """

    def __init__(self, showcase: Showcase):
        """Initialize with the showcase helper.

        Parameters
        ----------
        showcase : Showcase
            Required. Shared save/log/display helper.
        """
        self.showcase = showcase

    def run(self, raw_text: str) -> Tuple[pd.DataFrame, list]:
        """Parse layout predictions and save a table plus a label plot.

        Parameters
        ----------
        raw_text : str
            Required. Raw decoded model output.

        Returns
        -------
        Tuple[pd.DataFrame, list]
            Prediction dataframe and the raw regex matches.
        """
        matches, _, _ = uocr.re_match(raw_text)
        rows = []
        for _full, label, box in matches:
            try:
                coords = eval(box)  # noqa: S307 - model's own output format
                if coords and isinstance(coords[0], (int, float)):
                    coords = [coords]
            except Exception:  # noqa: BLE001
                coords = []
            for coord in coords or [[]]:
                rows.append(
                    {
                        "label": label.strip(),
                        "box": box,
                        "x1": coord[0] if len(coord) == 4 else None,
                        "y1": coord[1] if len(coord) == 4 else None,
                        "x2": coord[2] if len(coord) == 4 else None,
                        "y2": coord[3] if len(coord) == 4 else None,
                    }
                )
        df = pd.DataFrame(rows, columns=["label", "box", "x1", "y1", "x2", "y2"])
        self.showcase.save_table(df, "04_layout_predictions.csv")

        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                data=df,
                y="label",
                order=df["label"].value_counts().index,
                color="steelblue",
                ax=ax,
            )
            ax.set_title("Layout label distribution (BASE mode, page 1)")
            ax.set_xlabel("count")
            self.showcase.save_figure(fig, "04_label_distribution.png")
        logger.info(
            "[LayoutPredictionParser] parsed %d boxes across %d labels",
            len(df),
            df["label"].nunique() if not df.empty else 0,
        )
        return df, matches


layout_parser = LayoutPredictionParser(showcase)
layout_df, layout_matches = layout_parser.run(raw_output)
logger.info("Outputs: layout_df.shape=%s", layout_df.shape)
assert len(layout_df) > 0


# %% [markdown]
# ### Showcase: Boxed Result Image
# Draw the parsed boxes on the input page, mirroring `result_with_boxes.jpg`.


# %%
class BoxedResultShowcase:
    """Render parsed layout boxes onto the input page.

    Parameters
    ----------
    showcase : Showcase
        Required. Shared save/log/display helper.
    """

    def __init__(self, showcase: Showcase):
        """Initialize with the showcase helper.

        Parameters
        ----------
        showcase : Showcase
            Required. Shared save/log/display helper.
        """
        self.showcase = showcase

    def run(self, image: Image.Image, matches: list) -> Path:
        """Draw boxes and save the boxed result image.

        Parameters
        ----------
        image : Image.Image
            Required. Raw input page image.
        matches : list
            Required. Regex matches from the layout parser.

        Returns
        -------
        Path
            Path to the boxed result image.
        """
        result = uocr.process_image_with_refs(
            image.copy(), matches, str(self.showcase.dir)
        )
        path = self.showcase._save("05_result_with_boxes.jpg", lambda p: result.save(p))
        display(IPImage(filename=str(path)))
        logger.info("[BoxedResultShowcase] saved %s", path)
        return path


boxed_showcase = BoxedResultShowcase(showcase)
boxed_path = boxed_showcase.run(input_image, layout_matches)
logger.info("Outputs: boxed_path=%s", boxed_path)
assert boxed_path.exists()


# %% [markdown]
# ## Inspect Outputs
# List saved result files and preview text contents.


# %%
class OutputInspector:
    """Inspect generated output files."""

    TEXT_EXTS = {".txt", ".md", ".mmd", ".json"}

    def show(self, root: Path) -> None:
        """Print file list and text previews for a result directory.

        Parameters
        ----------
        root : Path
            Required. Directory to inspect.
        """
        logger.info("[OutputInspector] --- %s ---", root)
        if not root.is_dir():
            logger.info("[OutputInspector] (no output directory found)")
            return
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                fp = Path(dirpath) / fn
                size = fp.stat().st_size
                logger.info("[OutputInspector]  %s  (%s bytes)", fp, size)
                if Path(fn).suffix.lower() in self.TEXT_EXTS:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                    preview = content[:1500]
                    logger.info("[OutputInspector]  %s", "-" * 60)
                    for line in preview.splitlines():
                        logger.info("[OutputInspector]  | %s", line)
                    if len(content) > 1500:
                        logger.info(
                            "[OutputInspector]  | ... [%s more chars]",
                            len(content) - 1500,
                        )
                    logger.info("[OutputInspector]  %s", "-" * 60)


inspector = OutputInspector()
for run in CONFIG.runs:
    run_dir = CONFIG.out_dir / run.name
    inspector.show(run_dir)


# %% [markdown]
# ## Final Validation
# Verify all expected artifacts exist.

# %%
for run in CONFIG.runs:
    run_dir = CONFIG.out_dir / run.name
    assert run_dir.exists(), f"Missing run directory: {run_dir}"
    if run.run_gundam:
        assert (run_dir / "single_gundam").exists()
    if run.run_base:
        assert (run_dir / "single_base").exists()
    if run.run_multi:
        assert (run_dir / "multi_page").exists()
logger.info("[Validation] All expected artifacts exist")

for fname in (
    "00_input_page.png",
    "01_preprocessed_global_view.png",
    "01_preprocessed_tensor.png",
    "02_tokenized_prompt.json",
    "02_formatted_prompt.txt",
    "03_raw_generated_text.txt",
    "04_layout_predictions.csv",
    "04_label_distribution.png",
    "05_result_with_boxes.jpg",
):
    path = CONFIG.out_dir / "intermediate" / fname
    assert path.exists() and path.stat().st_size > 0, f"Missing intermediate: {path}"
    logger.info("[Validation] Found intermediate %s", path)
logger.info("[Validation] All intermediate artifacts exist and are non-empty")


# %%
logger.info("\nCheat sheet")
logger.info("GUNDAM mode: image_size=640, crop_mode=True")
logger.info("BASE mode: image_size=1024, crop_mode=False")
logger.info("Multi-page mode: image_size=1024, ngram_window=1024")
logger.info("Long outputs: keep max_length=32768 and no_repeat_ngram_size=35")
logger.info("Change CONFIG.runs or CONFIG.source to try other configurations")


# %% [markdown]
# ## Display Results
# Render the generated result markdown and boxed images inline when running in a notebook.

# %%
try:
    for run in CONFIG.runs:
        run_dir = CONFIG.out_dir / run.name
        for mode in ("single_gundam", "single_base", "multi_page"):
            mode_dir = run_dir / mode
            if not mode_dir.exists():
                continue
            result_md = mode_dir / "result.md"
            box_images = sorted(mode_dir.glob("result_with_boxes*.jpg"))
            if not result_md.exists() and not box_images:
                continue
            display(Markdown(f"### {run.name} / {mode}"))
            if result_md.exists():
                display(
                    Markdown(result_md.read_text(encoding="utf-8", errors="replace"))
                )
            for img_path in box_images:
                display(IPImage(filename=str(img_path)))
except Exception as exc:  # noqa: BLE001
    logger.info("Display skipped (not running in a notebook frontend): %s", exc)
