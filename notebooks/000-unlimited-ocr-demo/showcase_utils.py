"""Shared save + log + display helpers for the Unlimited-OCR demo notebooks.

Every notebook in ``notebooks/000-unlimited-ocr-demo/`` uses the :class:`Showcase`
helper to persist intermediate artifacts under
``outputs/<date>/000-unlimited-ocr-demo/<notebook-name>/intermediate/``, log a
shape/dtype summary, and render displayable artifacts inline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image as IPImage
from IPython.display import Markdown, display
from PIL import Image


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
