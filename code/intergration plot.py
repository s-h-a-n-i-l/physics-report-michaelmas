from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / (
    "data/initial set up intergrator test 6.2.1 and gain calc data/"
    "initial set up intergrator test 6.2.1 and gain calc data_01.csv"
)
PLOT_PATH = ROOT / "code/integrator_channels.png"


def load_trace(path: Path) -> pd.DataFrame:
    """Load oscilloscope traces while dropping the units row."""
    df = pd.read_csv(path, skiprows=[1])
    return df.dropna()


def peak_to_peak(series: pd.Series) -> float:
    return float(series.max() - series.min())


def main() -> None:
    df = load_trace(DATA_FILE)

    # Create two subplots (axes) sharing the same time axis
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6)
    )

    time = df["Time"]
    ch_a = df["Channel A"]
    ch_b = df["Channel B"]

    # Top axis: Channel A (input)
    ax_top.plot(time, ch_a, label="Channel A (input)")
    ax_top.set_ylabel("Voltage (V)")
    ax_top.set_title("Integrator Input (Channel A)")
    ax_top.legend()
    ax_top.grid(True, linestyle="--", alpha=0.4)

    # Bottom axis: Channel B (output)
    ax_bottom.plot(time, ch_b, label="Channel B (output)")
    ax_bottom.set_xlabel("Time (ms)")
    ax_bottom.set_ylabel("Voltage (V)")
    ax_bottom.set_title("Integrator Output (Channel B)")
    ax_bottom.legend()
    ax_bottom.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)

    # --- Gain calculation using peak-to-peak values ---
    vpp_a = peak_to_peak(ch_a)
    vpp_b = peak_to_peak(ch_b)

    # Gain = |Vout| / |Vin| using peak-to-peak amplitudes
    gain = abs(vpp_b / vpp_a) if vpp_a else float("nan")

    print(f"Channel A Vpp: {vpp_a:.3f} V")
    print(f"Channel B Vpp: {vpp_b:.3f} V")
    print(f"Integrator gain (|B|/|A|): {gain:.3f}")


if __name__ == "__main__":
    main()
