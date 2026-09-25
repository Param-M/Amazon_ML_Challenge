"""README figures (run after the pipeline; needs matplotlib).

    python -m ber.figures [out_dir]      # default: <repo>/docs/assets

1. house_number_shift  - share of candidate pairs that are true matches, by the
                         signed house-number difference (candidate - Source 1)
2. modifier_words      - per word added to the Source 1 name: how often the
                         house number still agrees (label-free) vs how often
                         the pair is a true match (labels)
Each figure is written in a light and a dark variant, plus the plotted data as CSV.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from . import features as F  # noqa: E402
from .config import ROOT, work  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", base="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", base="#383835", s1="#3987e5", s2="#d95926"),
}
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
PX = 1 / 96  # one CSS pixel in inches


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["base"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelsize=10, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _titles(fig, t, title, subtitle):
    fig.text(0.04, 0.955, title, color=t["ink"], fontsize=14.5, fontweight="bold", va="top")
    fig.text(0.04, 0.885, subtitle, color=t["ink2"], fontsize=10.5, va="top")


def _rounded_columns(ax, xs, hs, width, color, fig):
    """Columns with a 4px rounded data-end and a square baseline."""
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = 4 * PX * (x1 - x0) / bbox.width
    ry = 4 * PX * (y1 - y0) / bbox.height
    for x, h in zip(xs, hs):
        if h <= 0:
            continue
        left = x - width / 2
        r = min(rx, width / 2)
        ax.add_patch(FancyBboxPatch((left, 0), width, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=ry / r, facecolor=color, edgecolor="none"))
        if h > ry:
            ax.add_patch(Rectangle((left, 0), width, h - ry, facecolor=color, edgecolor="none"))


# ------------------------------------------------------------------ data

def shift_data(lo: int = -10, hi: int = 22) -> pl.DataFrame:
    R = F.load_records("train").select("a_hnum")
    C = pl.read_parquet(work("train_cands.parquet"), columns=["qr", "tr", "y"])
    h1 = R["a_hnum"][C["qr"].to_numpy()].cast(pl.Int64, strict=False)
    h2 = R["a_hnum"][C["tr"].to_numpy()].cast(pl.Int64, strict=False)
    d = (C.with_columns(shift=(h2 - h1)).filter(pl.col("shift").is_between(lo, hi))
         .group_by("shift").agg(pairs=pl.len(), true_matches=pl.col("y").sum())
         .with_columns(pct_true=100 * pl.col("true_matches") / pl.col("pairs")).sort("shift"))
    return d


def modifier_data(min_pairs: int = 300) -> pl.DataFrame:
    R = F.load_records("train").select("country", "n_core", "a_hnum")
    S = pl.read_parquet(work("train_stage_a_scores.parquet"), columns=["qr", "tr", "y"])
    parts = []
    for s in range(0, S.height, 10_000_000):
        c = S.slice(s, 10_000_000)
        q, t = R[c["qr"].to_numpy()], R[c["tr"].to_numpy()]
        d = pl.DataFrame({"country": q["country"], "nt1": q["n_core"].str.split(" "),
                          "nt2": t["n_core"].str.split(" "), "h1": q["a_hnum"], "h2": t["a_hnum"],
                          "y": c["y"]}).filter((pl.col("h1") != "") & (pl.col("h2") != ""))
        d = d.with_columns(extra=pl.col("nt2").list.set_difference(pl.col("nt1")),
                           missing=pl.col("nt1").list.set_difference(pl.col("nt2")))
        parts.append(d.filter((pl.col("extra").list.len() == 1) & (pl.col("missing").list.len() == 0))
                     .select("country", pl.col("extra").list.first().alias("word"),
                             (pl.col("h1") == pl.col("h2")).alias("heq"), "y"))
    d = pl.concat(parts)
    return (d.group_by("country", "word")
            .agg(pairs=pl.len(), pct_same_number=100 * pl.col("heq").mean(),
                 pct_true=100 * pl.col("y").cast(pl.Float64).mean())
            .filter(pl.col("pairs") >= min_pairs).sort("pairs", descending=True))


# ------------------------------------------------------------------ plots

def plot_shift(d: pl.DataFrame, out: Path, theme: str):
    t = THEMES[theme]
    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.15)
    _style(ax, t)
    xs, hs = d["shift"].to_numpy(), d["pct_true"].to_numpy()
    ax.set_xlim(xs.min() - 0.7, xs.max() + 0.7)
    ax.set_ylim(0, 100)
    _rounded_columns(ax, xs, hs, 0.62, t["s1"], fig)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:+d}" if x else "0" for x in xs], fontsize=8)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("House-number difference, candidate − Source 1", color=t["ink2"], fontsize=10, labelpad=8)
    same = d.filter(pl.col("shift") == 0)["pct_true"][0]
    ax.annotate(f"same number: {same:.0f}% true matches", xy=(0, same), xytext=(1.2, same + 2),
                color=t["ink2"], fontsize=10, va="center")
    low = d.filter((pl.col("shift") > 0) & (pl.col("pct_true") < 5))["shift"].to_list()
    ax.text(3.0, 44, "Look-alike businesses: number moved up by "
            + ", ".join(f"{x:+d}" for x in low) + "\n→ at most 3% of these pairs are true matches",
            color=t["ink2"], fontsize=10, va="bottom", linespacing=1.5)
    _titles(fig, t, "Look-alike businesses move the house number up by a few units",
            "Share of stage A candidate pairs that are true matches, by signed house-number difference (train)")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


# (country, word to point at, label text, label position in % units)
NOTES = (("US", "holdings", "+ holdings, + group, + southside … (US)", (3.0, 9.0)),
         ("India", "enterprises", "+ enterprises, + exports, + overseas … (India)", (13.5, 4.5)),
         ("US", "services", "+ services (US)", (18.5, 12.5)),
         ("India", "center", "+ center (India)", (25.5, 22.5)),
         ("India", "sri", "+ sri (India)", (20.5, 37.5)))


def plot_modifiers(d: pl.DataFrame, out: Path, theme: str, notes=NOTES):
    t = THEMES[theme]
    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.085, right=0.985, top=0.8, bottom=0.12)
    _style(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8, linestyle="-")
    colors = {"US": t["s1"], "India": t["s2"]}
    for country in ("US", "India"):
        s = d.filter(pl.col("country") == country)
        ax.scatter(s["pct_same_number"], s["pct_true"], s=46, color=colors[country],
                   edgecolors=t["surface"], linewidths=1.4, alpha=0.9, label=country, zorder=3)
    for country, word, text, pos in notes:
        r = d.filter((pl.col("country") == country) & (pl.col("word") == word))
        if r.height == 0:
            continue
        x, y = r["pct_same_number"][0], r["pct_true"][0]
        ax.annotate(text, xy=(x, y), xytext=pos, color=t["ink"], fontsize=10, va="center",
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8, shrinkA=2, shrinkB=4),
                    zorder=4)
    corr = np.corrcoef(d["pct_same_number"].to_numpy(), d["pct_true"].to_numpy())[0, 1]
    ax.text(0.985, 0.04, f"r = {corr:.2f} across {d.height} words (≥ 300 pairs each)",
            transform=ax.transAxes, ha="right", color=t["ink2"], fontsize=9.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Pairs whose house number still agrees (computed without labels)",
                  color=t["ink2"], fontsize=10, labelpad=8)
    ax.set_ylabel("Pairs that are true matches", color=t["ink2"], fontsize=10, labelpad=8)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10, handletextpad=0.3)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    _titles(fig, t, "Decoy words give themselves away: they almost never keep the address",
            "One dot per word added to a Source 1 name; train pairs with both house numbers present")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def main(out_dir=None):
    out = Path(out_dir) if out_dir else ROOT.parents[1] / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    s = shift_data()
    s.write_csv(out / "house_number_shift.csv")
    m = modifier_data()
    m.write_csv(out / "modifier_words.csv")
    for theme in ("light", "dark"):
        plot_shift(s, out / f"house_number_shift_{theme}.png", theme)
        plot_modifiers(m, out / f"modifier_words_{theme}.png", theme)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
