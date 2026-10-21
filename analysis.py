"""Exploratory analysis of the Palmer Penguins dataset.

Research question: do bill dimensions and body mass separate the three
penguin species, and how much overlap remains between Adelie and Chinstrap?

Usage:  python analysis.py
Outputs: fig_bill_dimensions.png, fig_body_mass.png,
         fig_flipper_vs_mass.png, summary_stats.csv
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "penguins.csv"

SPECIES_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
COLORS = {"Adelie": "#4C72B0", "Chinstrap": "#DD8452", "Gentoo": "#55A868"}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the penguin measurements and drop rows missing key variables."""
    df = pd.read_csv(path)
    key_columns = [
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    clean = df.dropna(subset=key_columns).copy()
    clean["species"] = pd.Categorical(
        clean["species"], categories=SPECIES_ORDER, ordered=True
    )
    return clean


def confidence_ellipse(x, y, ax, n_std: float = 1.5, **kwargs):
    """Draw an n-sigma covariance ellipse for a set of points."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse_radius_x = np.sqrt(1 + pearson)
    ellipse_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse(
        (0, 0),
        width=2 * ellipse_radius_x,
        height=2 * ellipse_radius_y,
        **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (
        matplotlib.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Median measurements per species, plus sample size."""
    return (
        df.groupby("species", observed=True)
        .agg(
            n=("body_mass_g", "size"),
            bill_length_mm=("bill_length_mm", "median"),
            bill_depth_mm=("bill_depth_mm", "median"),
            flipper_length_mm=("flipper_length_mm", "median"),
            body_mass_g=("body_mass_g", "median"),
        )
        .round(1)
    )


def nearest_centroid_loo_accuracy(df: pd.DataFrame) -> float:
    """Leave-one-out accuracy of a nearest-centroid classifier on bill shape."""
    features = ["bill_length_mm", "bill_depth_mm"]
    X = df[features].to_numpy(dtype=float)
    y = df["species"].astype(str).to_numpy()

    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    correct = 0
    for i in range(len(Z)):
        keep = np.ones(len(Z), dtype=bool)
        keep[i] = False
        centroids = {
            species: Z[keep & (y == species)].mean(axis=0)
            for species in SPECIES_ORDER
        }
        distances = {
            species: np.linalg.norm(Z[i] - centroid)
            for species, centroid in centroids.items()
        }
        prediction = min(distances, key=distances.get)
        correct += int(prediction == y[i])
    return correct / len(Z)


def plot_bill_dimensions(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="bill_length_mm",
        y="bill_depth_mm",
        hue="species",
        hue_order=SPECIES_ORDER,
        palette=COLORS,
        s=80,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
        legend=False,
    )
    for species in SPECIES_ORDER:
        subset = df[df["species"] == species]
        confidence_ellipse(
            subset["bill_length_mm"].to_numpy(),
            subset["bill_depth_mm"].to_numpy(),
            ax,
            n_std=1.5,
            facecolor=COLORS[species],
            edgecolor=COLORS[species],
            alpha=0.12,
            linewidth=2,
        )
        ax.scatter(
            subset["bill_length_mm"].median(),
            subset["bill_depth_mm"].median(),
            marker="X",
            s=220,
            color=COLORS[species],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            species,
            (subset["bill_length_mm"].median(), subset["bill_depth_mm"].median()),
            textcoords="offset points",
            xytext=(12, -14),
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )
    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Bill depth (mm)")
    ax.set_title("Bill shape separates the three species", pad=12)
    ax.text(
        0.01,
        0.99,
        "Crosses mark species medians; shading shows 1.5 SD ellipses",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="#555555",
    )
    fig.savefig(out_path)
    plt.close(fig)


def plot_body_mass(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.boxplot(
        data=df,
        x="species",
        y="body_mass_g",
        order=SPECIES_ORDER,
        hue="species",
        hue_order=SPECIES_ORDER,
        palette=COLORS,
        width=0.45,
        linewidth=1.2,
        fliersize=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="species",
        y="body_mass_g",
        order=SPECIES_ORDER,
        hue="species",
        hue_order=SPECIES_ORDER,
        palette=COLORS,
        size=4,
        alpha=0.45,
        jitter=0.18,
        legend=False,
        ax=ax,
    )
    medians = df.groupby("species", observed=True)["body_mass_g"].median()
    ax.set_xticks(
        range(len(SPECIES_ORDER)),
        labels=[
            f"{species}\nmedian {medians[species]:,.0f} g"
            for species in SPECIES_ORDER
        ],
        fontsize=13,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Body mass (g)")
    ax.set_title("Gentoo penguins are markedly heavier", pad=12)
    ax.set_ylim(2500, 6600)
    ax.set_xlim(-0.6, 2.6)
    fig.savefig(out_path)
    plt.close(fig)


def plot_flipper_vs_mass(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.scatterplot(
        data=df,
        x="flipper_length_mm",
        y="body_mass_g",
        hue="species",
        hue_order=SPECIES_ORDER,
        palette=COLORS,
        s=75,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    for species in SPECIES_ORDER:
        subset = df[df["species"] == species]
        slope, intercept = np.polyfit(
            subset["flipper_length_mm"], subset["body_mass_g"], 1
        )
        x_values = np.linspace(
            subset["flipper_length_mm"].min(),
            subset["flipper_length_mm"].max(),
            50,
        )
        ax.plot(x_values, intercept + slope * x_values, color=COLORS[species], lw=2)

    r_value = df["flipper_length_mm"].corr(df["body_mass_g"])
    ax.text(
        0.03,
        0.96,
        f"Pearson r = {r_value:.2f} (all species)",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "pad": 6},
    )
    ax.set_xlabel("Flipper length (mm)")
    ax.set_ylabel("Body mass (g)")
    ax.set_title("Flipper length tracks body mass within each species", pad=12)
    ax.legend(
        title="Species",
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    df = load_data()
    stats = summarise(df)

    stats.to_csv(BASE_DIR / "summary_stats.csv")
    plot_bill_dimensions(df, BASE_DIR / "fig_bill_dimensions.png")
    plot_body_mass(df, BASE_DIR / "fig_body_mass.png")
    plot_flipper_vs_mass(df, BASE_DIR / "fig_flipper_vs_mass.png")

    accuracy = nearest_centroid_loo_accuracy(df)
    r_value = df["flipper_length_mm"].corr(df["body_mass_g"])
    adelie = df[df["species"] == "Adelie"]
    chinstrap = df[df["species"] == "Chinstrap"]
    gentoo = df[df["species"] == "Gentoo"]

    print(f"Clean sample: {len(df)} penguins across {df['species'].nunique()} species")
    print()
    print("Median measurements by species")
    print(stats.to_string())
    print()
    print("Key findings")
    print(
        "- Gentoo median body mass: "
        f"{gentoo['body_mass_g'].median():,.0f} g, versus "
        f"{adelie['body_mass_g'].median():,.0f} g (Adelie) and "
        f"{chinstrap['body_mass_g'].median():,.0f} g (Chinstrap)."
    )
    print(
        "- Adelie vs Chinstrap bill length range: "
        f"{adelie['bill_length_mm'].min():.1f}-{adelie['bill_length_mm'].max():.1f} mm "
        f"vs {chinstrap['bill_length_mm'].min():.1f}-"
        f"{chinstrap['bill_length_mm'].max():.1f} mm."
    )
    print(
        "- Median bill depth: "
        f"Adelie {adelie['bill_depth_mm'].median():.1f} mm, "
        f"Chinstrap {chinstrap['bill_depth_mm'].median():.1f} mm, "
        f"Gentoo {gentoo['bill_depth_mm'].median():.1f} mm."
    )
    print(
        "- Leave-one-out nearest-centroid accuracy using bill length and depth: "
        f"{accuracy:.1%}."
    )
    print(f"- Pearson r between flipper length and body mass: {r_value:.2f}.")
    print(
        "- Missing values removed: "
        f"{pd.read_csv(DATA_PATH).shape[0] - len(df)} rows."
    )


if __name__ == "__main__":
    main()
