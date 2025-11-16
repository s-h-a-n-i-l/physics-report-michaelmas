from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "straight line wave from air.csv"
OUTPUT_FIG = BASE_DIR / "straight_line_air_fit.png"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the straight-line data and return a clean DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path, skiprows=[1])
    df = df.dropna(subset=["Channel A", "Channel B"])
    return df


def fit_line(x: np.ndarray, y: np.ndarray):
    """
    Return slope/intercept and a slope uncertainty derived from the scatter of points.

    The uncertainty is the standard error of the slope:
        σ_m = sqrt( (1/(n-2)) * Σ(residual^2) / Σ(x - x̄)^2 )
    """
    slope, intercept = np.polyfit(x, y, deg=1)

    y_pred = slope * x + intercept
    residuals = y - y_pred
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points to estimate slope uncertainty.")

    ss_res = np.sum(residuals**2)
    ss_x = np.sum((x - x.mean()) ** 2)
    variance = ss_res / (n - 2)
    slope_unc = np.sqrt(variance / ss_x)
    return float(slope), float(intercept), float(slope_unc)


def main():
    df = load_data(DATA_FILE)
    x = df["Channel A"].to_numpy()
    y = df["Channel B"].to_numpy() / 1000  # Convert mV to V

    slope, intercept, slope_unc = fit_line(x, y)
    x_intercept = -intercept / slope if not np.isclose(slope, 0.0) else np.nan

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=1, color="tab:blue", label="Measured data")
    plt.plot(x_fit, y_fit, color="tab:red", label=f"Fit: y = {slope:.4f}x + {intercept:.4f}")
    plt.xlabel("Channel A (V)")
    plt.ylabel("Channel B (V)")
    plt.title("Straight Line Data (Air)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG)

    print(f"Slope (gradient): {slope:.6f} ± {slope_unc:.6f} V/V")
    print(f"Y-intercept: {intercept:.6f} V")
    print(f"X-intercept: {x_intercept:.6f} V")
    print(f"Plot saved to {OUTPUT_FIG}")
    


if __name__ == "__main__":
    main()
