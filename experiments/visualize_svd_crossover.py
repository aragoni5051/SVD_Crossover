from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from spectral_ga.crossover import CrossoverConfig, anchor_layer_mode_crossover
from spectral_ga.network import NetworkConfig, SpectralNetwork
from spectral_ga.spectral import reconstruct_layer


def unit_circle(n: int = 400) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def make_parent(seed: int, angle: float, scales: tuple[float, float]) -> SpectralNetwork:
    """Create a 2D one-layer network with a controlled SVD-like geometry."""
    c, s = np.cos(angle), np.sin(angle)
    left = np.array([[c, -s], [s, c]], dtype=float)
    right_angle = -0.55 * angle + 0.35
    cr, sr = np.cos(right_angle), np.sin(right_angle)
    right = np.array([[cr, -sr], [sr, cr]], dtype=float)
    sigma = np.diag(np.asarray(scales, dtype=float))
    weight = left @ sigma @ right.T

    net = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 2], r_max=2, seed=seed))
    from spectral_ga.spectral import decompose_dense

    gene = net.layers[0].gene
    net.layers[0] = decompose_dense(weight, r_max=2, keep_residual=False)
    net.layers[0].gene = gene
    return net


def mode_source(child_layer, anchor_layer, donor_layer, mode_index: int, anchor_name: str, donor_name: str) -> str:
    if mode_index >= child_layer.rank:
        return "-"
    from_anchor = (
        mode_index < anchor_layer.rank
        and np.allclose(child_layer.alpha[mode_index], anchor_layer.alpha[mode_index])
        and np.allclose(child_layer.u[:, mode_index], anchor_layer.u[:, mode_index])
        and np.allclose(child_layer.v[:, mode_index], anchor_layer.v[:, mode_index])
    )
    from_donor = (
        mode_index < donor_layer.rank
        and np.allclose(child_layer.alpha[mode_index], donor_layer.alpha[mode_index])
        and np.allclose(child_layer.u[:, mode_index], donor_layer.u[:, mode_index])
        and np.allclose(child_layer.v[:, mode_index], donor_layer.v[:, mode_index])
    )
    if from_donor:
        return donor_name
    if from_anchor:
        return anchor_name
    return "mixed/zero"


def plot_transform(ax, weight: np.ndarray, layer, title: str, color: str) -> None:
    circle = unit_circle()
    mapped = circle @ weight.T
    ax.plot(circle[:, 0], circle[:, 1], color="0.75", linewidth=1.0, linestyle="--", label="unit circle")
    ax.plot(mapped[:, 0], mapped[:, 1], color=color, linewidth=2.2, label="W x")

    origin = np.zeros(2)
    for i in range(layer.rank):
        v = layer.v[:, i]
        u = layer.u[:, i] * layer.alpha[i]
        ax.arrow(origin[0], origin[1], v[0], v[1], color="tab:purple", width=0.008, length_includes_head=True, alpha=0.85)
        ax.arrow(origin[0], origin[1], u[0], u[1], color="tab:orange", width=0.008, length_includes_head=True, alpha=0.85)
        ax.text(v[0] * 1.08, v[1] * 1.08, f"v{i}", color="tab:purple", fontsize=8)
        ax.text(u[0] * 1.03, u[1] * 1.03, f"a{i}u{i}", color="tab:orange", fontsize=8)

    ax.axhline(0, color="0.9", linewidth=0.8)
    ax.axvline(0, color="0.9", linewidth=0.8)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize two actual SVD-mode crossover children.")
    parser.add_argument("--output", type=Path, default=Path("results/visualizations/svd_crossover_demo.png"))
    parser.add_argument("--method", choices=["half_rank", "single_point", "uniform"], default="half_rank")
    parser.add_argument("--uniform-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for this visualization: {exc}") from exc

    parent_a = make_parent(seed=args.seed, angle=0.35, scales=(2.8, 0.65))
    parent_b = make_parent(seed=args.seed + 1, angle=1.15, scales=(1.55, 1.05))
    config = CrossoverConfig(method=args.method, uniform_prob=args.uniform_prob)
    child_ab = anchor_layer_mode_crossover(
        parent_a,
        parent_b,
        layer_index=0,
        config=config,
        rng=np.random.default_rng(args.seed + 2),
    )
    child_ba = anchor_layer_mode_crossover(
        parent_b,
        parent_a,
        layer_index=0,
        config=config,
        rng=np.random.default_rng(args.seed + 2),
    )

    layer_a = parent_a.layers[0]
    layer_b = parent_b.layers[0]
    layer_ab = child_ab.layers[0]
    layer_ba = child_ba.layers[0]
    weight_a = reconstruct_layer(layer_a)
    weight_b = reconstruct_layer(layer_b)
    weight_ab = reconstruct_layer(layer_ab)
    weight_ba = reconstruct_layer(layer_ba)

    sources_ab = [mode_source(layer_ab, layer_a, layer_b, i, "A", "B") for i in range(layer_ab.rank)]
    sources_ba = [mode_source(layer_ba, layer_b, layer_a, i, "B", "A") for i in range(layer_ba.rank)]
    copied_ab = [str(i) for i, source in enumerate(sources_ab) if source == "B"]
    kept_ab = [str(i) for i, source in enumerate(sources_ab) if source == "A"]
    copied_ba = [str(i) for i, source in enumerate(sources_ba) if source == "A"]
    kept_ba = [str(i) for i, source in enumerate(sources_ba) if source == "B"]

    fig = plt.figure(figsize=(16, 8.5))
    fig.suptitle("Actual SVD-Mode Crossover", fontsize=22, y=0.96)

    ax_a = fig.add_subplot(2, 4, 1)
    ax_b = fig.add_subplot(2, 4, 2)
    ax_ab = fig.add_subplot(2, 4, 3)
    ax_ba = fig.add_subplot(2, 4, 4)
    plot_transform(ax_a, weight_a, layer_a, "Parent A", "tab:blue")
    plot_transform(ax_b, weight_b, layer_b, "Parent B", "tab:red")
    plot_transform(ax_ab, weight_ab, layer_ab, "Child 1: A anchor + B modes", "tab:green")
    plot_transform(ax_ba, weight_ba, layer_ba, "Child 2: B anchor + A modes", "tab:cyan")

    ax_text = fig.add_subplot(2, 4, 5)
    ax_text.axis("off")
    ax_text.text(0.02, 0.84, r"$W = U\,diag(\alpha)\,V^T$", fontsize=22)
    ax_text.text(0.02, 0.66, "Crossover unit:", fontsize=13, weight="bold")
    ax_text.text(0.08, 0.54, r"one mode package $(u_i, \alpha_i, v_i)$", fontsize=13)
    ax_text.text(0.02, 0.36, "Child 1:", fontsize=13, weight="bold")
    ax_text.text(0.08, 0.26, f"kept from A: {', '.join(kept_ab) if kept_ab else 'none'}", fontsize=12)
    ax_text.text(0.08, 0.16, f"copied from B: {', '.join(copied_ab) if copied_ab else 'none'}", fontsize=12)
    ax_text.text(0.02, 0.04, f"Child 2 keeps B: {', '.join(kept_ba) if kept_ba else 'none'}; copies A: {', '.join(copied_ba) if copied_ba else 'none'}", fontsize=12)

    ax_table = fig.add_subplot(2, 4, 6)
    ax_table.axis("off")
    rows = []
    max_rank = max(layer_ab.rank, layer_ba.rank)
    for i in range(max_rank):
        alpha_a = layer_a.alpha[i] if i < layer_a.rank else 0.0
        alpha_b = layer_b.alpha[i] if i < layer_b.rank else 0.0
        source_ab = sources_ab[i] if i < len(sources_ab) else "-"
        source_ba = sources_ba[i] if i < len(sources_ba) else "-"
        rows.append([i, source_ab, source_ba, f"{alpha_a:.3f}", f"{alpha_b:.3f}"])
    table = ax_table.table(
        cellText=rows,
        colLabels=["mode", "child 1", "child 2", "alpha A", "alpha B"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)
    ax_table.set_title("Mode inheritance")

    ax_heat_ab = fig.add_subplot(2, 4, 7)
    diff_ab = weight_ab - weight_a
    im_ab = ax_heat_ab.imshow(diff_ab, cmap="coolwarm")
    ax_heat_ab.set_title("Child 1 - Parent A dense W")
    ax_heat_ab.set_xticks(range(diff_ab.shape[1]))
    ax_heat_ab.set_yticks(range(diff_ab.shape[0]))
    fig.colorbar(im_ab, ax=ax_heat_ab, fraction=0.046, pad=0.04)

    ax_heat_ba = fig.add_subplot(2, 4, 8)
    diff_ba = weight_ba - weight_b
    im_ba = ax_heat_ba.imshow(diff_ba, cmap="coolwarm")
    ax_heat_ba.set_title("Child 2 - Parent B dense W")
    ax_heat_ba.set_xticks(range(diff_ba.shape[1]))
    ax_heat_ba.set_yticks(range(diff_ba.shape[0]))
    fig.colorbar(im_ba, ax=ax_heat_ba, fraction=0.046, pad=0.04)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
