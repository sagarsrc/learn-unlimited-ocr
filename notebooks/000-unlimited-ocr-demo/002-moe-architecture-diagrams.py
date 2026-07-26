# %%
import datetime
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Paths and hyperparameters for Unlimited-OCR MoE diagrams.

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
    out_dir=Path(f"outputs/{TODAY}/unlimited-ocr-demo-moe-diagrams"),
    model_name="baidu/Unlimited-OCR",
    sample_text=(
        "The quick brown fox jumps over the lazy dog. "
        "Document understanding with mixture-of-experts models."
    ),
)
logger.info("Config: model=%s out_dir=%s", CONFIG.model_name, CONFIG.out_dir)


# %% [markdown]
# ## What is MoE in Unlimited-OCR?
# Most layers are **sparse Mixture-of-Experts (MoE)** layers. Instead of one big MLP
# for every token, a small router (gate) picks a few experts (here 6 out of 64)
# for each token. Only the chosen experts compute, so most weights stay idle.
# The first layer is dense (every token uses the same MLP).

# %% [markdown]
# ## Check Environment


# %%
class EnvironmentChecker:
    """Check GPU availability and select the inference dtype."""

    def __init__(self, config: Config):
        self.config = config

    def check(self) -> torch.dtype:
        """Assert CUDA is available and return the selected dtype."""
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
# ## Load Model and Capture Routing


# %%
class ModelLoader:
    """Load the Unlimited-OCR model and tokenizer."""

    def __init__(self, config: Config):
        self.config = config

    def load(self) -> Tuple[AutoTokenizer, AutoModel]:
        """Download and load the model."""
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


class MoEInspector:
    """Capture MoE routing decisions during a forward pass."""

    def __init__(self, model: AutoModel):
        self.model = model
        self.activations: Dict[str, Dict[str, torch.Tensor]] = {}

    def _make_hook(self, name: str) -> Callable:
        """Return a hook that stores top-k expert indices and weights."""

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
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run a forward pass and return captured activations."""
        self.activations = {}
        with torch.no_grad():
            _ = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
        logger.info(
            "[MoEInspector] Captured activations from %s MoE layers",
            len(self.activations),
        )
        return self.activations


model_loader = ModelLoader(CONFIG)
tokenizer, model = model_loader.load()

moe_inspector = MoEInspector(model)
moe_inspector.register_hooks()
inputs = tokenizer(CONFIG.sample_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
ACTIVATIONS = moe_inspector.run(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)
TOKENS = [
    t.replace("\uff5c", "|")
    for t in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
]
N_ROUTED = getattr(model.config, "n_routed_experts", 64)
TOP_K = getattr(model.config, "num_experts_per_tok", 6)
LAYERS = sorted(ACTIVATIONS.keys(), key=lambda x: int(x.split(".")[2]))
logger.info(
    "Outputs: %s tokens, %s MoE layers, %s experts, top-k=%s",
    len(TOKENS),
    len(LAYERS),
    N_ROUTED,
    TOP_K,
)
assert len(ACTIVATIONS) > 0


# %% [markdown]
# ## Diagram 1: Whole Model Stack
# Show all 12 layers. Green = dense MLP. Purple = sparse MoE layer.


# %%
class WholeModelStackDiagram:
    """Draw a high-level stack of dense vs. MoE layers."""

    def __init__(self, total_layers: int, first_dense_layers: int):
        self.total_layers = total_layers
        self.first_dense_layers = first_dense_layers

    def draw(self, output_path: Path) -> None:
        """Save the model-stack diagram."""
        fig, ax = plt.subplots(figsize=(4, 10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, self.total_layers + 1)
        ax.axis("off")
        ax.set_title("Unlimited-OCR Decoder Stack\n(purple = MoE, green = dense)")

        for i in range(self.total_layers):
            is_dense = i < self.first_dense_layers
            color = "#8fd694" if is_dense else "#c49ae9"
            label = "Dense MLP" if is_dense else f"MoE (64 experts, top-{TOP_K})"
            y = self.total_layers - i
            rect = FancyBboxPatch(
                (0.15, y - 0.4),
                0.7,
                0.75,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
            )
            ax.add_patch(rect)
            ax.text(0.5, y, f"Layer {i}\n{label}", ha="center", va="center")
            if i < self.total_layers - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.45),
                    xytext=(0.5, y - 0.05),
                    arrowprops=dict(arrowstyle="->", color="black"),
                )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[WholeModelStackDiagram] Saved %s", output_path)


stack_diagram = WholeModelStackDiagram(
    total_layers=12,
    first_dense_layers=getattr(model.config, "first_k_dense_replace", 1),
)
CONFIG.out_dir.mkdir(parents=True, exist_ok=True)
stack_diagram.draw(CONFIG.out_dir / "00_model_stack.png")
assert (CONFIG.out_dir / "00_model_stack.png").exists()


# %% [markdown]
# ## Diagram 2: Inside a Dense Layer vs. an MoE Layer
# Compare a dense MLP (all parameters used) with an MoE layer (gate picks experts).


# %%
class LayerComparisonDiagram:
    """Draw side-by-side dense and MoE layer internals."""

    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k

    def _draw_dense(self, ax: plt.Axes) -> None:
        """Draw a dense MLP block."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title("Dense Layer (Layer 0)")

        ax.add_patch(
            FancyBboxPatch(
                (2, 6.5),
                6,
                1.5,
                boxstyle="round,pad=0.1",
                facecolor="lightblue",
                edgecolor="black",
            )
        )
        ax.text(5, 7.25, "Self-Attention", ha="center", va="center")
        ax.annotate(
            "",
            xy=(5, 6.4),
            xytext=(5, 8.2),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        ax.add_patch(
            FancyBboxPatch(
                (2, 3),
                6,
                2,
                boxstyle="round,pad=0.1",
                facecolor="#8fd694",
                edgecolor="black",
            )
        )
        ax.text(
            5,
            4,
            "Dense MLP\n(every token uses\nthe same weights)",
            ha="center",
            va="center",
        )
        ax.annotate(
            "",
            xy=(5, 2.8),
            xytext=(5, 5.1),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        ax.text(5, 1.8, "Output", ha="center", va="center")

    def _draw_moe(self, ax: plt.Axes) -> None:
        """Draw an MoE layer block."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title("MoE Layer (Layers 1-11)")

        ax.add_patch(
            FancyBboxPatch(
                (2, 8.5),
                6,
                1,
                boxstyle="round,pad=0.1",
                facecolor="lightblue",
                edgecolor="black",
            )
        )
        ax.text(5, 9, "Self-Attention", ha="center", va="center")
        ax.annotate(
            "",
            xy=(5, 8.4),
            xytext=(5, 9.7),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        ax.add_patch(
            FancyBboxPatch(
                (2.5, 6.5),
                5,
                1.2,
                boxstyle="round,pad=0.1",
                facecolor="lightyellow",
                edgecolor="black",
            )
        )
        ax.text(
            5,
            7.1,
            f"Router / Gate\npicks top-{self.top_k} experts",
            ha="center",
            va="center",
        )
        ax.annotate(
            "",
            xy=(5, 6.4),
            xytext=(5, 7.7),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        # expert grid
        cols = 16
        rows = 4
        box_w = 6.0 / cols
        box_h = 0.75
        selected = {2, 15, 23, 31, 45, 58}
        for i in range(self.num_experts):
            row = i // cols
            col = i % cols
            x = 2 + col * box_w
            y = 4.8 - row * box_h
            color = "orange" if i in selected else "lightgray"
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    box_w * 0.9,
                    box_h * 0.8,
                    facecolor=color,
                    edgecolor="black",
                )
            )
        ax.text(
            5,
            5.5,
            "Expert bank (64 experts; orange = active for this token)",
            ha="center",
            va="center",
        )
        ax.annotate(
            "",
            xy=(5, 2.3),
            xytext=(5, 4.8),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        ax.add_patch(
            FancyBboxPatch(
                (2, 0.8),
                6,
                1.1,
                boxstyle="round,pad=0.1",
                facecolor="lightcoral",
                edgecolor="black",
            )
        )
        ax.text(
            5,
            1.35,
            "Weighted Sum\n(only active experts contribute)",
            ha="center",
            va="center",
        )

    def draw(self, output_path: Path) -> None:
        """Save side-by-side comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        self._draw_dense(axes[0])
        self._draw_moe(axes[1])
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[LayerComparisonDiagram] Saved %s", output_path)


comparison_diagram = LayerComparisonDiagram(num_experts=N_ROUTED, top_k=TOP_K)
comparison_diagram.draw(CONFIG.out_dir / "01_dense_vs_moe_layer.png")
assert (CONFIG.out_dir / "01_dense_vs_moe_layer.png").exists()


# %% [markdown]
# ## Diagram 3: One Token's Journey Through the MoE Layers
# Pick a token and show which experts are activated at each MoE layer.


# %%
class TokenJourneyDiagram:
    """Draw the active experts for a chosen token across MoE layers."""

    def __init__(
        self,
        activations: Dict[str, Dict[str, torch.Tensor]],
        layers: List[str],
        tokens: List[str],
        top_k: int,
    ):
        self.activations = activations
        self.layers = layers
        self.tokens = tokens
        self.top_k = top_k

    def draw(self, token_index: int, output_path: Path) -> None:
        """Save the token-journey diagram for one token.

        Parameters
        ----------
        token_index : int
            Required. Index of the token to visualize.
        output_path : Path
            Required. Path for the output PNG.
        """
        token = self.tokens[token_index]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, len(self.layers) + 1)
        ax.set_ylim(0, self.top_k + 1)
        ax.set_title(f"Token journey: '{token}'\nexperts selected at each MoE layer")
        ax.set_xlabel("MoE Layer")
        ax.set_ylabel("Selected expert rank (1 = highest gate score)")
        ax.set_xticks(range(1, len(self.layers) + 1))
        ax.set_xticklabels([f"L{int(name.split('.')[2])}" for name in self.layers])
        ax.set_yticks(range(1, self.top_k + 1))
        ax.invert_yaxis()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for row, layer_name in enumerate(self.layers, start=1):
            topk_idx = self.activations[layer_name]["topk_idx"][token_index]
            topk_weight = self.activations[layer_name]["topk_weight"][token_index]
            for rank, (expert, weight) in enumerate(
                zip(topk_idx, topk_weight), start=1
            ):
                ax.scatter(row, rank, s=200 + float(weight) * 2000, c="orange")
                ax.text(
                    row,
                    rank,
                    str(int(expert)),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[TokenJourneyDiagram] Saved %s", output_path)


journey_diagram = TokenJourneyDiagram(
    activations=ACTIVATIONS, layers=LAYERS, tokens=TOKENS, top_k=TOP_K
)
# Choose the token "Document" if it exists, otherwise the first non-BOS token.
TARGET_TOKEN_INDEX = next((i for i, t in enumerate(TOKENS) if "Document" in t), 1)
journey_diagram.draw(
    TARGET_TOKEN_INDEX,
    CONFIG.out_dir / "02_token_journey.png",
)
assert (CONFIG.out_dir / "02_token_journey.png").exists()


# %% [markdown]
# ## Diagram 4: Aggregate Expert Usage
# Bar chart showing how often each expert is selected across all tokens and layers.


# %%
class ExpertUsageChart:
    """Draw aggregate expert usage as a bar chart."""

    def __init__(
        self,
        activations: Dict[str, Dict[str, torch.Tensor]],
        num_experts: int,
    ):
        self.activations = activations
        self.num_experts = num_experts

    def draw(self, output_path: Path) -> None:
        """Save the aggregate usage bar chart."""
        usage = np.zeros(self.num_experts, dtype=np.float32)
        for layer_name in self.activations:
            topk_idx = self.activations[layer_name]["topk_idx"]
            topk_weight = self.activations[layer_name]["topk_weight"]
            for token_i in range(topk_idx.shape[0]):
                for k in range(topk_idx.shape[1]):
                    expert = int(topk_idx[token_i, k])
                    usage[expert] += float(topk_weight[token_i, k])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(self.num_experts), usage)
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Total routed weight")
        ax.set_title("Aggregate expert usage across all tokens and MoE layers")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[ExpertUsageChart] Saved %s", output_path)


usage_chart = ExpertUsageChart(ACTIVATIONS, N_ROUTED)
usage_chart.draw(CONFIG.out_dir / "03_aggregate_expert_usage.png")
assert (CONFIG.out_dir / "03_aggregate_expert_usage.png").exists()


# %% [markdown]
# ## Final Validation
# Verify all diagrams were created.

# %%
expected = [
    CONFIG.out_dir / "00_model_stack.png",
    CONFIG.out_dir / "01_dense_vs_moe_layer.png",
    CONFIG.out_dir / "02_token_journey.png",
    CONFIG.out_dir / "03_aggregate_expert_usage.png",
]
for path in expected:
    assert path.exists(), f"Missing artifact: {path}"
logger.info("[Validation] All diagrams saved")


# %%
logger.info("\nKey takeaways")
logger.info("- Layer 0 is a dense MLP; every token uses the same weights.")
logger.info(
    "- Layers 1-11 are MoE: each token selects %s of %s experts.",
    TOP_K,
    N_ROUTED,
)
logger.info(
    "- Only the selected experts compute for that token, so most parameters stay idle."
)
logger.info(
    "- The aggregate chart shows that some experts are reused more often than others."
)
