# %%
import datetime
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Paths and hyperparameters for inspecting Unlimited-OCR MoE routing.

    Parameters
    ----------
    work_dir : Path
        Required. Project root used to resolve relative paths.
    out_dir : Path
        Required. Directory for all generated artifacts.
    model_name : str
        Required. Hugging Face model identifier.
    sample_text : str
        Required. Prompt text used to drive the forward pass.
    dtype : torch.dtype | None
        Optional. Selected torch dtype, set at runtime.
    """

    work_dir: Path
    out_dir: Path
    model_name: str
    sample_text: str
    dtype: torch.dtype | None = None

    def replace(self, **kwargs: object) -> "Config":
        """Return a new Config with the given fields replaced."""
        return replace(self, **kwargs)


TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
CONFIG = Config(
    work_dir=Path.cwd(),
    out_dir=Path(f"outputs/{TODAY}/unlimited-ocr-demo-moe-architecture-inspection"),
    model_name="baidu/Unlimited-OCR",
    sample_text=(
        "The quick brown fox jumps over the lazy dog. "
        "Document understanding with mixture-of-experts models "
        "enables efficient long-horizon parsing."
    ),
)
logger.info("Config: model=%s out_dir=%s", CONFIG.model_name, CONFIG.out_dir)


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
# Load the Unlimited-OCR model and tokenizer onto the GPU.


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
# ## Architecture Summary
# Print the MoE configuration and identify which layers are sparse vs. dense.


# %%
class ArchitectureSummarizer:
    """Summarize the MoE architecture of the loaded model.

    Parameters
    ----------
    model : AutoModel
        Required. Loaded model.
    """

    def __init__(self, model: AutoModel):
        """Initialize with the loaded model.

        Parameters
        ----------
        model : AutoModel
            Required. Loaded model.
        """
        self.model = model

    def summarize(self) -> Dict[str, Any]:
        """Return a dictionary with MoE architecture facts.

        Returns
        -------
        Dict[str, Any]
            Architecture summary.
        """
        cfg = self.model.config
        num_layers = len(self.model.model.layers)
        moe_layers: List[int] = []
        dense_layers: List[int] = []
        for i, layer in enumerate(self.model.model.layers):
            mlp_type = type(layer.mlp).__name__
            if mlp_type == "DeepseekV2MoE":
                moe_layers.append(i)
            else:
                dense_layers.append(i)
        summary = {
            "total_layers": num_layers,
            "moe_layers": moe_layers,
            "dense_layers": dense_layers,
            "n_routed_experts": getattr(cfg, "n_routed_experts", None),
            "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
            "topk_method": getattr(cfg, "topk_method", None),
            "first_k_dense_replace": getattr(cfg, "first_k_dense_replace", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
        }
        logger.info("[ArchitectureSummarizer] Summary:")
        for key, value in summary.items():
            logger.info("  %s: %s", key, value)
        return summary


summarizer = ArchitectureSummarizer(model)
ARCH_SUMMARY = summarizer.summarize()
assert ARCH_SUMMARY["n_routed_experts"] is not None
assert ARCH_SUMMARY["num_experts_per_tok"] is not None


# %% [markdown]
# ## Inspect MoE Routing
# Register forward hooks on the gating modules and capture expert choices.


# %%
class MoEInspector:
    """Capture MoE routing decisions during a forward pass.

    Parameters
    ----------
    model : AutoModel
        Required. Loaded model.
    """

    def __init__(self, model: AutoModel):
        """Initialize with the loaded model.

        Parameters
        ----------
        model : AutoModel
            Required. Loaded model.
        """
        self.model = model
        self.activations: Dict[str, Dict[str, torch.Tensor]] = {}

    def _make_hook(self, name: str):
        """Return a hook that stores top-k expert indices and weights.

        Parameters
        ----------
        name : str
            Required. Module name used as the activation key.

        Returns
        -------
        Callable
            Forward hook function.
        """

        def hook(module, input, output):
            topk_idx, topk_weight, _aux_loss = output
            self.activations[name] = {
                "topk_idx": topk_idx.detach().cpu(),
                "topk_weight": topk_weight.detach().cpu(),
            }

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on every MoEGate module."""
        self.activations = {}
        for name, module in self.model.named_modules():
            if type(module).__name__ == "MoEGate":
                module.register_forward_hook(self._make_hook(name))

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run a forward pass and return captured activations.

        Parameters
        ----------
        input_ids : torch.Tensor
            Required. Token IDs on the model device.
        attention_mask : torch.Tensor
            Required. Attention mask on the model device.

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            Mapping from layer name to captured activation tensors.
        """
        self.activations = {}
        with torch.no_grad():
            _ = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
        logger.info(
            "[MoEInspector] Captured activations from %s MoE layers",
            len(self.activations),
        )
        return self.activations


moe_inspector = MoEInspector(model)
moe_inspector.register_hooks()
inputs = tokenizer(CONFIG.sample_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
ACTIVATIONS = moe_inspector.run(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)
assert len(ACTIVATIONS) > 0, "No MoE activations captured"
logger.info(
    "Outputs: seq_len=%s tokens=%s",
    inputs["input_ids"].shape[1],
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
)


# %% [markdown]
# ## Visualize Activations
# Plot heat maps of active experts and their weights per token.


# %%
class MoEVisualizer:
    """Visualize captured MoE routing decisions.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Required. Loaded tokenizer.
    input_ids : torch.Tensor
        Required. Token IDs used for the forward pass.
    activations : Dict[str, Dict[str, torch.Tensor]]
        Required. Captured MoE activations.
    num_experts : int
        Required. Total number of routed experts.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        input_ids: torch.Tensor,
        activations: Dict[str, Dict[str, torch.Tensor]],
        num_experts: int,
    ):
        """Initialize with tokenizer, input IDs, activations, and expert count."""
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.activations = activations
        self.num_experts = num_experts

    def _clean_tokens(self) -> List[str]:
        """Return printable token labels for the x-axis."""
        tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[0])
        return [t.replace("\u65372", "|") for t in tokens]

    def _dense_weights(self, layer_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Build a dense [seq_len, num_experts] weight array from sparse top-k.

        Parameters
        ----------
        layer_name : str
            Required. Name of the MoE layer to convert.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Dense weights and binary active mask.
        """
        act = self.activations[layer_name]
        topk_idx = act["topk_idx"]
        topk_weight = act["topk_weight"]
        seq_len = topk_idx.shape[0]
        dense = np.zeros((seq_len, self.num_experts), dtype=np.float32)
        for token_i in range(seq_len):
            for k in range(topk_idx.shape[1]):
                expert = int(topk_idx[token_i, k])
                dense[token_i, expert] = float(topk_weight[token_i, k])
        return dense, (dense > 0).astype(np.float32)

    def _configure_axis(self, ax: plt.Axes) -> None:
        """Set readable tick labels on a heatmap axis."""
        ax.set_yticks(np.arange(0, self.num_experts, 10))
        ax.set_yticklabels(np.arange(0, self.num_experts, 10))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_horizontalalignment("right")

    def plot_layer_heatmaps(self, output_path: Path) -> None:
        """Save a combined heatmap of expert weights for every MoE layer.

        Parameters
        ----------
        output_path : Path
            Required. Path for the output PNG.
        """
        layer_names = sorted(
            self.activations.keys(),
            key=lambda x: int(x.split(".")[2]),
        )
        tokens = self._clean_tokens()
        n_layers = len(layer_names)
        fig, axes = plt.subplots(
            n_layers, 1, figsize=(14, 1.8 * n_layers), squeeze=False
        )
        for ax, layer_name in zip(axes.flat, layer_names):
            dense, _ = self._dense_weights(layer_name)
            sns.heatmap(
                dense.T,
                ax=ax,
                cmap="viridis",
                cbar=True,
                xticklabels=tokens,
                yticklabels=False,
                vmin=0,
                vmax=dense.max(),
            )
            self._configure_axis(ax)
            ax.set_title(layer_name)
            ax.set_xlabel("Token")
            ax.set_ylabel("Expert")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[MoEVisualizer] Saved layer heatmaps to %s", output_path)

    def plot_binary_active_heatmap(self, output_path: Path) -> None:
        """Save a binary heatmap showing which experts are active per token.

        Parameters
        ----------
        output_path : Path
            Required. Path for the output PNG.
        """
        layer_names = sorted(
            self.activations.keys(),
            key=lambda x: int(x.split(".")[2]),
        )
        tokens = self._clean_tokens()
        fig, axes = plt.subplots(
            len(layer_names), 1, figsize=(14, 1.8 * len(layer_names)), squeeze=False
        )
        for ax, layer_name in zip(axes.flat, layer_names):
            _, binary = self._dense_weights(layer_name)
            sns.heatmap(
                binary.T,
                ax=ax,
                cmap="Greys",
                cbar=False,
                xticklabels=tokens,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
            self._configure_axis(ax)
            ax.set_title(f"{layer_name} — active experts")
            ax.set_xlabel("Token")
            ax.set_ylabel("Expert")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[MoEVisualizer] Saved binary active heatmap to %s", output_path)

    def plot_aggregate_usage(self, output_path: Path) -> None:
        """Save a bar chart of total expert usage across all captured layers.

        Parameters
        ----------
        output_path : Path
            Required. Path for the output PNG.
        """
        usage = np.zeros(self.num_experts, dtype=np.float32)
        for layer_name in self.activations:
            dense, _ = self._dense_weights(layer_name)
            usage += dense.sum(axis=0)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(self.num_experts), usage)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Total routed weight")
        ax.set_title("Aggregate expert usage across all MoE layers")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[MoEVisualizer] Saved aggregate usage chart to %s", output_path)


visualizer = MoEVisualizer(
    tokenizer,
    inputs["input_ids"],
    ACTIVATIONS,
    ARCH_SUMMARY["n_routed_experts"],
)
CONFIG.out_dir.mkdir(parents=True, exist_ok=True)
visualizer.plot_layer_heatmaps(CONFIG.out_dir / "moe_weights_heatmap.png")
visualizer.plot_binary_active_heatmap(CONFIG.out_dir / "moe_active_heatmap.png")
visualizer.plot_aggregate_usage(CONFIG.out_dir / "moe_aggregate_usage.png")


# %% [markdown]
# ## Final Validation
# Verify that all expected artifacts exist.

# %%
expected = [
    CONFIG.out_dir / "moe_weights_heatmap.png",
    CONFIG.out_dir / "moe_active_heatmap.png",
    CONFIG.out_dir / "moe_aggregate_usage.png",
]
for path in expected:
    assert path.exists(), f"Missing artifact: {path}"
logger.info("[Validation] All expected artifacts exist")


# %%
logger.info("\nCheat sheet")
logger.info("Total routed experts: %s", ARCH_SUMMARY["n_routed_experts"])
logger.info("Experts chosen per token (top-k): %s", ARCH_SUMMARY["num_experts_per_tok"])
logger.info("MoE layers: %s", ARCH_SUMMARY["moe_layers"])
logger.info("Dense layers: %s", ARCH_SUMMARY["dense_layers"])
