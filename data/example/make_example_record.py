"""Regenerate the synthetic format example; this is not research data."""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).with_name("example_record.csv")
DT_SECONDS = 0.005
DURATION_SECONDS = 6.0

time_s = np.arange(0.0, DURATION_SECONDS, DT_SECONDS)
envelope = np.exp(-0.5 * ((time_s - 2.6) / 0.75) ** 2)
h1_g = 0.16 * envelope * np.sin(2.0 * np.pi * 3.2 * time_s)
h2_g = 0.11 * envelope * np.sin(2.0 * np.pi * 2.3 * time_s + 0.7)
v_g = 0.06 * envelope * np.sin(2.0 * np.pi * 5.1 * time_s + 0.2)

pd.DataFrame(
    {
        "time_s": time_s,
        "h1_g": h1_g,
        "h2_g": h2_g,
        "v_g": v_g,
    }
).to_csv(OUTPUT_PATH, index=False)

print(f"Wrote {OUTPUT_PATH} with {len(time_s)} synthetic samples.")

