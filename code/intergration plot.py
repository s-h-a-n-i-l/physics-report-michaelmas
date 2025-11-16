from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "initial set up intergrator test 6.2.1 and gain calc data"
CSV_PATTERN = "initial set up intergrator test 6.2.1 and gain calc data_*.csv"
OUTPUT_FIG = BASE_DIR / "integration_plot.png"


def load_channel_data(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate every capture file in the provided directory."""
    csv_paths = sorted(data_dir.glob(CSV_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching pattern {CSV_PATTERN!r} in {data_dir}")

    frames = [
        pd.read_csv(path, skiprows=[1]).dropna(subset=["Time", "Channel A", "Channel B"])
        for path in csv_paths
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("Time")
    return data


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the cumulative integral using the trapezoidal rule."""
    dx = np.diff(x)
    running = np.cumsum((y[:-1] + y[1:]) * 0.5 * dx)
    return np.concatenate(([0.0], running))


# def scale_factor(reference: np.ndarray, target: np.ndarray) -> float:
#     """
#     Compute least-squares alpha such that alpha * reference ≈ target.
#     """
#     denom = np.dot(reference, reference)
#     if np.isclose(denom, 0.0):
#         raise ValueError("Integral trace is all zeros; cannot determine scale factor.")
#     return float(np.dot(reference, target) / denom)


def peak_to_peak_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Compute (peak-to-peak of A) / (peak-to-peak of B)."""
    ptp_a = np.ptp(a)
    ptp_b = np.ptp(b)
    if np.isclose(ptp_b, 0.0):
        raise ValueError("Channel B peak-to-peak is zero; cannot form ratio.")
    return ptp_a / ptp_b


def main():
    df = load_channel_data(DATA_DIR)
    time_ms = df["Time"].to_numpy()
    channel_a = df["Channel A"].to_numpy()
    channel_b = df["Channel B"].to_numpy()

    time_s = time_ms / 1000.0
    window_mask = (time_s >= 0.1) & (time_s <= 0.2)
    if not np.any(window_mask):
        raise ValueError("No samples fall within 0.1–0.2 seconds in the dataset.")

    time_s = time_s[window_mask]
    channel_a = channel_a[window_mask]
    channel_b = channel_b[window_mask]

    cumulative_integral = cumulative_trapezoid(time_s, channel_a)
    # alpha = scale_factor(cumulative_integral, channel_b)
    # scaled_integral = alpha * cumulative_integral
    ptp_ratio = peak_to_peak_ratio(channel_a, channel_b)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(time_s, channel_a, color="tab:blue", label="Channel A (input)")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Channel A Input Signal (0.1–0.2 s)")
    axes[0].legend()

    axes[1].plot(time_s, cumulative_integral, color="tab:green", label="∫ Channel A d(time)")
    axes[1].set_ylabel("Integral (V·s)")
    axes[1].set_title("Cumulative Integral (Trapezoidal Rule)")
    axes[1].legend()

    axes[2].plot(time_s, channel_b, color="tab:orange", label="Channel B")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].set_title("Channel B Output")
    axes[2].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG)

    print(f"Peak-to-peak ratio (Channel A / Channel B): {ptp_ratio:.6e}")
    print(f"Saved figure to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
