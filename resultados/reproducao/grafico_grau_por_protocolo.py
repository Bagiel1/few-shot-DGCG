import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fewshot-grau")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "resultados/analise_experimentos_20261026/fontes/melhores_graus_por_protocolo_metricas.csv"
OUTPUT = ROOT / "resultados/graficos_para_aprovacao"

ORDER = ["2:4", "4:6", "6:8", "8:12"]
LABELS = {"2:4": "2–4", "4:6": "4–6", "6:8": "6–8", "8:12": "8–12"}
COLORS = {
    "2:4": "#3B6EA8",
    "4:6": "#E58C3A",
    "6:8": "#5A9A74",
    "8:12": "#8B6FA8",
}


def main() -> None:
    data = pd.read_csv(SOURCE)
    data = data[data["metodo"].eq("dgcg")].copy()
    protocols = [(shot, query) for shot in (1, 5, 10) for query in (1, 5, 10)]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), sharey=True)

    for ax, way in zip(axes, (2, 3, 5)):
        subset = data[data["n_way"].eq(way)]
        counts = pd.DataFrame(0, index=range(len(protocols)), columns=ORDER, dtype=int)
        node_counts = []
        for index, (shot, query) in enumerate(protocols):
            group = subset[subset["n_shot"].eq(shot) & subset["n_query"].eq(query)]
            frequencies = group["alvo"].value_counts()
            for target in ORDER:
                counts.loc[index, target] = int(frequencies.get(target, 0))
            node_counts.append(way * (shot + query))

        bottom = pd.Series(0, index=counts.index, dtype=float)
        x = range(len(protocols))
        for target in ORDER:
            values = counts[target]
            bars = ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.72,
                color=COLORS[target],
                edgecolor="white",
                linewidth=0.7,
                label=LABELS[target],
            )
            for bar, value, base in zip(bars, values, bottom):
                if value >= 2:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + value / 2,
                        str(int(value)),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                        fontweight="bold",
                    )
            bottom += values

        for index, nodes in enumerate(node_counts):
            ax.text(index, 10.18, f"N={nodes}", ha="center", va="bottom", fontsize=7, color="#4A4A4A")

        ax.set_title(f"{way}-way")
        ax.set_xticks(list(x), [f"S{shot}\nQ{query}" for shot, query in protocols])
        ax.set_ylim(0, 11.05)
        ax.set_yticks(range(0, 11, 2))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Shot e query por classe")

    axes[0].set_ylabel("Bases que selecionaram a faixa (de 10)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Faixa de grau-alvo",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        frameon=False,
    )
    fig.suptitle(
        "Faixa de grau do DGCG selecionada por protocolo",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "Seleção pela maior acurácia de validação em cada base Micro · seed 7 · cada barra soma 10 bases",
        ha="center",
        fontsize=9.5,
        color="#3F3F3F",
    )
    fig.text(
        0.5,
        0.012,
        "N = way × (shot + query). Os números dentro das barras indicam quantas bases escolheram cada faixa; o teste não participa da seleção.",
        ha="center",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.735, wspace=0.12)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "grau_dgcg_por_protocolo.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / "grau_dgcg_por_protocolo.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
