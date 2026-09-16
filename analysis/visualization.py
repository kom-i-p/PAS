from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "data" / "visualizations"

SUBJECT_MATRIX_FILE = (
    PROCESSED_DIR / "subject_reachability_matrix.csv"
)

DISTRICT_MATRIX_FILE = (
    PROCESSED_DIR / "district_reachability_matrix.csv"
)


def load_matrix(path):
    return pd.read_csv(
        path,
        index_col=0,
        encoding="utf-8-sig",
    )


def plot_matrix(matrix, title, output_file):
    figure_size = max(10, len(matrix) * 0.18)

    figure, axis = plt.subplots(
        figsize=(figure_size, figure_size)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        aspect="equal",
    )

    axis.set_title(title)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_yticks(range(len(matrix.index)))

    axis.set_xticklabels(
        matrix.columns,
        rotation=90,
        fontsize=6,
    )
    axis.set_yticklabels(
        matrix.index,
        fontsize=6,
    )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.04, shrink=0.7)
    colorbar.ax.tick_params(labelsize=6)
    colorbar.set_label("Количество переходов")

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Visualization: {output_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_matrix = load_matrix(SUBJECT_MATRIX_FILE)
    plot_matrix(
        subject_matrix,
        "Матрица топологической достижимости субъектов РФ",
        OUTPUT_DIR / "subject_reachability_matrix.png",
    )

    district_matrix = load_matrix(DISTRICT_MATRIX_FILE)
    plot_matrix(
        district_matrix,
        "Матрица топологической достижимости федеральных округов",
        OUTPUT_DIR / "district_reachability_matrix.png",
    )


if __name__ == "__main__":
    main()