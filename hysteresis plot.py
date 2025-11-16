"""
Generate annotated B–H hysteresis plots for every dataset in `hysteresis data/`.

Implementation follows the TODO in this file:
1. Load each oscilloscope capture (1 s of data) and isolate a single 0.02 s
   window representing one hysteresis loop (20 ms centred around t≈0).
2. Convert Channel A → H (A/m) and Channel B → B (T) using coil parameters.
3. Plot B vs H, compute loop area via `np.trapz`, capture the maximum |dB/dH|,
   and measure the gradients of the six furthest left/right points.
4. Repeat for every CSV in the `hysteresis data` tree, saving one PNG per file
   alongside a console summary so each figure is self-contained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

# Instrument / geometry constants (SI units)
nprim = 400
nsec = 500
Lprim = 0.042  # m
Lprim_error = 0.001  # m (unused but kept for record)
Ri = 32680
Ri_error = 10
C_val = 97.8e-9
C_val_error = 0.1e-9
R_prim = 2
mu_0 = 4e-7 * np.pi

# Cross-sectional areas (mm^2)
material_area_mm2 = {
    "transformer_iron": 2.4,
    "mild_steel": 7.65,
    "copper_alloy": 19.6,
}

# Optional measurement uncertainties (unused but preserved)
material_area_error_mm2 = {
    "transformer_iron": 0.016,
    "mild_steel": 0.02,
    "copper_alloy": 0.16,
}

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "hysteresis data"
PLOT_ROOT = BASE_DIR / "hysteresis_plots"
LOOP_DURATION_S = 0.02  # seconds of data (20 ms) per loop


def H_calc(v_primary: np.ndarray) -> np.ndarray:
    """Convert measured primary voltage to magnetising field H (A/m)."""
    return v_primary * (nprim / (Lprim * R_prim))


def B_calc(v_secondary: np.ndarray, material: str) -> np.ndarray:
    """Convert measured secondary voltage to flux density B (T)."""
    area_m2 = material_area_mm2[material] * 1e-6  # mm² → m²
    return (C_val * v_secondary * Ri) / (area_m2 * nsec)


def infer_material_from_path(csv_path: Path) -> str:
    """Infer material key from any part of the dataset path or filename."""
    lowered = str(csv_path).lower()
    for key in material_area_mm2:
        if key in lowered:
            return key
    raise ValueError(
        f"Unable to infer material for '{csv_path}'. "
        f"Expected one of {list(material_area_mm2)} to appear in the path or filename."
    )


def load_capture(csv_path: Path) -> pd.DataFrame:
    """Load a single hysteresis capture, normalising Channel A/B to volts."""
    units_map: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        headers = fh.readline().strip().split(",")
        unit_line = fh.readline().strip().split(",")
        for name, unit in zip(headers, unit_line):
            unit_clean = unit.strip().strip("()").lower()
            units_map[name.strip()] = unit_clean

    df = pd.read_csv(csv_path, skiprows=[1])
    df = df.dropna(subset=["Channel A", "Channel B"])
    df = df.astype(float)

    # Ensure Channel A/B are in volts regardless of original units.
    scale_map = {
        "v": 1.0,
        "mv": 1e-3,
        "uv": 1e-6,
        "kv": 1e3,
    }
    for col in ("Channel A", "Channel B"):
        unit = units_map.get(col, "").lower()
        factor = scale_map.get(unit, 1.0)
        df[col] *= factor

    time_col = df.columns[0]
    df = df.rename(columns={time_col: "Time_ms"})
    df = df.sort_values("Time_ms").reset_index(drop=True)
    return df


def extract_loop_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a 0.02 s window (20 ms) centred on the time closest to zero.

    Falls back to the entire capture if the dataset is shorter than the window.
    """
    time_ms = df["Time_ms"].to_numpy()
    if len(time_ms) == 0:
        return df

    center_idx = int(np.argmin(np.abs(time_ms)))
    center_time = time_ms[center_idx]
    half_window = (LOOP_DURATION_S * 1000) / 2.0  # convert to ms
    lower = center_time - half_window
    upper = center_time + half_window
    mask = (time_ms >= lower) & (time_ms <= upper)
    if not np.any(mask):
        return df
    return df.loc[mask].reset_index(drop=True)


def gradient_metrics(
    h: np.ndarray,
    b: np.ndarray,
    edge_points: int = 2,
) -> tuple[float, float, float, float]:
    """
    Return gradient metrics:
        - max_grad: maximum absolute |dB/dH| across the loop
        - left_grad: slope fitted through the `edge_points` smallest H samples
        - right_grad: slope fitted through the `edge_points` largest H samples
        - min_flat_grad: min absolute value between left/right slopes
    """
    if len(h) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    h_arr = np.asarray(h)
    b_arr = np.asarray(b)

    dh = np.gradient(h_arr)
    db = np.gradient(b_arr)
    mask = np.abs(dh) > 1e-12
    if np.any(mask):
        grads = np.abs(db[mask] / dh[mask])
        max_grad = float(np.max(grads))
    else:
        max_grad = float("nan")

    def slope_of_edge(select_high: bool) -> float:
        if edge_points < 2:
            return float("nan")
        order = np.argsort(h_arr)
        idx = order[-edge_points:] if select_high else order[:edge_points]
        idx = np.sort(idx)
        if len(idx) < 2:
            return float("nan")
        coeffs = np.polyfit(h_arr[idx], b_arr[idx], 1)
        return float(coeffs[0])

    left_grad = slope_of_edge(select_high=False)
    right_grad = slope_of_edge(select_high=True)

    grads = [g for g in (left_grad, right_grad) if not np.isnan(g)]
    min_flat_grad = float(np.min(np.abs(grads))) if grads else float("nan")

    return max_grad, left_grad, right_grad, min_flat_grad


def loop_statistics(h: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Collect loop metrics for annotation."""
    area = float(np.trapz(b, h))
    max_grad, left_grad, right_grad, min_flat_grad = gradient_metrics(h, b)
    return {
        "loop_area": area,
        "h_min": float(np.min(h)),
        "h_max": float(np.max(h)),
        "b_min": float(np.min(b)),
        "b_max": float(np.max(b)),
        "grad_max": max_grad,
        "grad_min": left_grad,
        "grad_right": right_grad,
        "grad_min_flat": min_flat_grad,
        'mur min': min_flat_grad/mu_0 if not np.isnan(min_flat_grad) else float('nan'),
        'mur max': max_grad/mu_0 if not np.isnan(max_grad) else float('nan'),
        "power_loss_per_unit_volume": abs(area*-50),
    }


def plot_loop(
    h: np.ndarray,
    b: np.ndarray,
    dataset: str,
    capture_name: str,
    stats: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot and save a single annotated hysteresis loop."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h, b, color="tab:blue", linewidth=1.2)

    ax.set_xlabel("H (A/m)")
    ax.set_ylabel("B (T)")
    ax.set_title(f" {capture_name[5:].replace('_', ' ')} hysteresis loop (20 ms segment)".title())
    ax.grid(alpha=0.3)
    formatter_x = ScalarFormatter(useOffset=False)
    formatter_x.set_scientific(False)
    formatter_y = ScalarFormatter(useOffset=False)
    formatter_y.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)

    annotation = (
        f"H range: {stats['h_min']:.2f} → {stats['h_max']:.2f} A/m\n"
        f"B range: {stats['b_min']:.3f} → {stats['b_max']:.3f} T\n"
        f"Loop area: {abs(stats['loop_area']):.3e} T·A/m\n"
        f"max μr: {stats['mur max']:.2f}\n"
        f"Min μr (ends): {stats['mur min']:.2f}"
        f"\nPower loss per unit volume: {stats['power_loss_per_unit_volume']:.3f} W/m³"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def iter_captures() -> List[Path]:
    """Yield every CSV capture under DATA_ROOT."""
    return sorted(DATA_ROOT.rglob("*.csv"))


def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_ROOT}")

    csv_files = iter_captures()
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {DATA_ROOT}")

    summary_rows = []

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(DATA_ROOT)
        dataset_label = rel_path.parent / rel_path.stem if rel_path.parent != Path(".") else Path(rel_path.stem)
        dataset_label_str = str(dataset_label)
        material_key = infer_material_from_path(csv_path)

        df = load_capture(csv_path)
        loop_df = extract_loop_window(df)

        h = H_calc(loop_df["Channel A"].to_numpy())
        b = B_calc(loop_df["Channel B"].to_numpy(), material_key)
        stats = loop_statistics(h, b)

        output_dir = PLOT_ROOT / rel_path.parent
        output_path = output_dir / f"{csv_path.stem}_hysteresis.png"
        plot_loop(h, b, dataset_label_str, csv_path.stem, stats, output_path)

        summary_rows.append(
            {
                "dataset_label": dataset_label_str,
                "capture": csv_path.stem,
                "material": material_key,
                "h_min_A_per_m": stats["h_min"],
                "h_max_A_per_m": stats["h_max"],
                "b_min_T": stats["b_min"],
                "b_max_T": stats["b_max"],
                "loop_area_TA_per_m": stats["loop_area"],
                "mu_r_min": stats["mur min"],
                "mu_r_max": stats["mur max"],
                "grad_max": stats["grad_max"],
                "grad_left": stats.get("grad_min", np.nan),
                "grad_right": stats.get("grad_right"),
                "grad_min_flat": stats["grad_min_flat"],
                "power_loss_W_per_m3": stats["power_loss_per_unit_volume"],
                "plot_path": str(output_path),
            }
        )

        print(
            f"{dataset_label_str}: "
            f"H [{stats['h_min']:.2f}, {stats['h_max']:.2f}] A/m | "
            f"B [{stats['b_min']:.3f}, {stats['b_max']:.3f}] T | "
            f"Area {stats['loop_area']:.3e} T·A/m | "
            f"max |dB/dH|={stats['grad_max']:.2e}, "
            f"min |.|={stats['grad_min_flat']:.2e}] "
            f"→ {output_path}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = PLOT_ROOT / "hysteresis_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary table to {summary_csv}")


if __name__ == "__main__":
    main()
