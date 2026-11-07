import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fewshot-grau-real")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "resultados/analise_experimentos_20261026/fontes/melhores_graus_por_protocolo_metricas.csv"
OUTPUT = ROOT / "resultados/graficos_para_aprovacao"


def main() -> None:
    data = pd.read_csv(SOURCE)
    ways = [2, 3, 5]
    shots = [1, 5, 10]
    queries = [1, 5, 10]
    summary = (
        data.groupby(["n_way", "n_shot", "n_query"], as_index=False)
        .agg(grau_real=("test_degree_mean", "mean"))
    )
    vmax = float(summary["grau_real"].max())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.15), constrained_layout=False)
    image = None
    for ax, way in zip(axes, ways):
        values = np.zeros((3, 3), dtype=float)
        for row, shot in enumerate(shots):
            for col, query in enumerate(queries):
                item = summary[
                    summary.n_way.eq(way)
                    & summary.n_shot.eq(shot)
                    & summary.n_query.eq(query)
                ].iloc[0]
                values[row, col] = item.grau_real

        image = ax.imshow(values, cmap="Blues", vmin=1, vmax=vmax, aspect="equal")
        ax.set_title(f"{way}-way", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(range(3), [str(value) for value in queries])
        ax.set_yticks(range(3), [str(value) for value in shots])
        ax.set_xlabel("Queries por classe")
        if way == 2:
            ax.set_ylabel("Shots por classe")

        threshold = 0.57 * vmax
        for row, shot in enumerate(shots):
            for col, query in enumerate(queries):
                nodes = way * (shot + query)
                value = values[row, col]
                color = "white" if value >= threshold else "#1E1E1E"
                ax.text(
                    col,
                    row - 0.08,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )
                ax.text(
                    col,
                    row + 0.22,
                    f"N={nodes}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=color,
                    alpha=0.92,
                )
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
    colorbar.set_label("Grau médio realizado")
    fig.suptitle(
        "Grau médio do DGCG em cada protocolo",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Faixa escolhida pela validação em cada base · valor médio no teste das 10 bases Micro · seed 7",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.025,
        "Cada célula é um protocolo. N indica o número de nós do episódio.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.9, bottom=0.16, top=0.83, wspace=0.28)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "grau_medio_realizado_por_protocolo.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / "grau_medio_realizado_por_protocolo.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
