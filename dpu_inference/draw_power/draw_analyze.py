#!/usr/bin/env python3
"""
KV260 功耗記錄分析工具
讀取 power_log.sh 產生的 CSV（uptime_s,power_uW,current_mA,voltage_mV），
畫出功耗曲線並輸出統計表。

用法範例：
  python3 power_analyze.py ip_power.csv ps_power.csv
  python3 power_analyze.py ip_power.csv ps_power.csv --labels IP PS --bin 0.5
  python3 power_analyze.py ip_power.csv ps_power.csv --idle idle.csv --fps 120 15
  python3 power_analyze.py ip_power.csv --trim 1.0 --no-show
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_csv(path, trim=0.0):
    df = pd.read_csv(path)
    df["t"] = df["uptime_s"] - df["uptime_s"].iloc[0]
    df["W"] = df["power_uW"] / 1e6
    if trim > 0:  # 去掉頭尾各 trim 秒（例如載入模型 / 結束階段）
        end = df["t"].iloc[-1]
        df = df[(df["t"] >= trim) & (df["t"] <= end - trim)].copy()
        df["t"] -= df["t"].iloc[0]
    return df.reset_index(drop=True)


def summarize(df, label, idle_mean=None, fps=None):
    w = df["W"]
    dur = df["t"].iloc[-1] - df["t"].iloc[0]
    energy = _trapz(w, df["t"])
    row = {
        "label": label,
        "duration_s": dur,
        "samples": len(df),
        "mean_W": w.mean(),
        "median_W": w.median(),
        "std_W": w.std(),
        "min_W": w.min(),
        "max_W": w.max(),
        "p5_W": w.quantile(0.05),
        "p95_W": w.quantile(0.95),
        "mean_mA": df["current_mA"].mean(),
        "mean_mV": df["voltage_mV"].mean(),
        "energy_J": energy,
    }
    if idle_mean is not None:
        row["dynamic_W"] = row["mean_W"] - idle_mean
        row["dynamic_energy_J"] = row["dynamic_W"] * dur
    if fps is not None:
        row["fps"] = fps
        row["fps_per_W"] = fps / row["mean_W"]
        row["mJ_per_frame"] = row["mean_W"] / fps * 1000
    return row


def binned(df, bin_s):
    if bin_s <= 0:
        return df["t"], df["W"]
    b = (df["t"] // bin_s) * bin_s
    g = df.groupby(b)["W"].mean()
    return g.index.values, g.values


def plot_curves(dfs, labels, bin_s, idle_mean, out_dir):
    n = len(dfs)
    ymin = min(d["W"].min() for d in dfs) - 0.5
    ymax = max(d["W"].max() for d in dfs) + 0.5
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1) 每個檔案各一張子圖
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for i, (df, lab) in enumerate(zip(dfs, labels)):
        ax = axes[i][0]
        c = colors[i % len(colors)]
        ax.plot(df["t"], df["W"], color=c, alpha=0.25, lw=0.8, label="raw")
        if bin_s > 0:
            x, y = binned(df, bin_s)
            ax.plot(x, y, color=c, lw=1.6, label=f"{bin_s:g}s avg")
        ax.axhline(df["W"].mean(), color="k", ls="--", lw=1,
                   label=f"mean {df['W'].mean():.2f} W")
        if idle_mean is not None:
            ax.axhline(idle_mean, color="gray", ls=":", lw=1,
                       label=f"idle {idle_mean:.2f} W")
        ax.set_title(f"{lab} power")
        ax.set_ylabel("Power (W)")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1][0].set_xlabel("Time (s)")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "power_curves.png")
    fig.savefig(p1, dpi=150)

    # 2) 疊圖比較 + 箱型圖
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4),
                                  gridspec_kw={"width_ratios": [3, 1]})
    for i, (df, lab) in enumerate(zip(dfs, labels)):
        x, y = binned(df, bin_s)
        a1.plot(x, y, lw=1.4, color=colors[i % len(colors)], label=lab)
    if idle_mean is not None:
        a1.axhline(idle_mean, color="gray", ls=":", lw=1, label="idle")
    a1.set_title("Power comparison")
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Power (W)")
    a1.grid(alpha=0.3)
    a1.legend()
    data = [d["W"] for d in dfs]
    try:
        a2.boxplot(data, tick_labels=labels, showmeans=True)  # matplotlib >= 3.9
    except TypeError:
        a2.boxplot(data, labels=labels, showmeans=True)       # 舊版
    a2.set_title("Distribution")
    a2.set_ylabel("Power (W)")
    a2.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "power_compare.png")
    fig2.savefig(p2, dpi=150)
    return [p1, p2]


def plot_table(summary, out_dir):
    cols = [c for c in summary.columns if c != "label"]
    cell = [[f"{v:.3f}" if isinstance(v, float) else str(v)
             for v in summary[c]] for c in cols]
    fig, ax = plt.subplots(figsize=(2.2 + 1.8 * len(summary), 0.35 * len(cols) + 0.8))
    ax.axis("off")
    tb = ax.table(cellText=cell, rowLabels=cols,
                  colLabels=list(summary["label"]), loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.3)
    ax.set_title("Power summary", pad=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "power_summary.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    return p


def main():
    ap = argparse.ArgumentParser(description="KV260 power log analyzer")
    ap.add_argument("files", nargs="+", help="power_log.sh 產生的 CSV")
    ap.add_argument("--labels", nargs="+", help="各檔案的名稱（預設用檔名）")
    ap.add_argument("--bin", type=float, default=0.5, help="平滑視窗秒數，0 = 不平滑")
    ap.add_argument("--idle", help="idle 基準 CSV，用來算動態功耗")
    ap.add_argument("--fps", nargs="+", type=float, help="各檔案對應的 FPS，用來算能效")
    ap.add_argument("--trim", type=float, default=0.0, help="去掉頭尾各幾秒")
    ap.add_argument("--out", default=".", help="輸出資料夾")
    ap.add_argument("--no-show", action="store_true", help="不跳出視窗，只存檔")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    os.makedirs(args.out, exist_ok=True)

    labels = args.labels or [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    if len(labels) != len(args.files):
        ap.error("--labels 數量要跟檔案數量一樣")
    if args.fps and len(args.fps) != len(args.files):
        ap.error("--fps 數量要跟檔案數量一樣")

    dfs = [load_csv(f, args.trim) for f in args.files]
    idle_mean = load_csv(args.idle)["W"].mean() if args.idle else None

    rows = [summarize(df, lab, idle_mean, args.fps[i] if args.fps else None)
            for i, (df, lab) in enumerate(zip(dfs, labels))]
    summary = pd.DataFrame(rows)

    # 終端機輸出
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(summary.set_index("label").T.to_string())
    if idle_mean is not None:
        print(f"\nidle mean = {idle_mean:.3f} W")

    csv_path = os.path.join(args.out, "power_summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    imgs = plot_curves(dfs, labels, args.bin, idle_mean, args.out)
    imgs.append(plot_table(summary, args.out))
    print("\n輸出檔案:", csv_path, *imgs, sep="\n  ")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()