# %%
import datetime
import logging
import os
import tempfile
import requests
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence, Tuple

import fitz
import matplotlib.pyplot as plt
import torch
from IPython.display import Markdown, Image as IPImage, display
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    runs=(
        RunConfig(
            name="first_4_all",
            max_pages=4,
            run_gundam=True,
            run_base=True,
            run_multi=True,
        ),
        RunConfig(
            name="first_1_compact",
            max_pages=1,
            run_gundam=True,
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
