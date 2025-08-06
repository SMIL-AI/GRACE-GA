"""
Enhanced Visitation-Weighted Gravity Accessibility (2019)
---------------------------------------------------------
Steps:
1. Load merged 2019 POI-CBG dataset
2. Compute visit shares (r_ij)
3. Apply Gaussian distance decay
4. Calculate accessibility scores A_i^G per CBG
5. Save results
"""

import os
import pickle
import pandas as pd
import numpy as np

# -------------------------
# Config
# -------------------------
BASE_PATH = "./sample_data"  # Change to your actual base path
MERGED_FILE = os.path.join(BASE_PATH, "merged_data_2019.pkl")
OUTPUT_FILE = os.path.join(BASE_PATH, "accessibility_data_2019.pkl")

# Gaussian decay parameters
D0 = 10000  # characteristic decay distance (meters)
EPSILON = 1e-6

# -------------------------
# Helper functions
# -------------------------
def gaussian_decay_modified(d, d0=D0, epsilon=EPSILON):
    """Gaussian distance decay with small epsilon to avoid div-by-zero."""
    return np.exp(-0.5 * (d / d0) ** 2) / (1 + epsilon)

# -------------------------
# Main processing
# -------------------------
if __name__ == "__main__":
    print("1. Loading merged 2019 dataset...")
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(f"File not found: {MERGED_FILE}")

    with open(MERGED_FILE, "rb") as f:
        df_acc = pickle.load(f)

    required_cols = ["CBG_ID", "PLACEKEY", "Distance", "RAW_VISIT_COUNTS", "Total Population", "Vij"]
    for col in required_cols:
        if col not in df_acc.columns:
            raise ValueError(f"Missing required column: {col}")

    print("2. Calculating visit shares r_ij...")
    df_acc["r_ij"] = df_acc["Vij"] / df_acc.groupby("PLACEKEY")["Vij"].transform("sum")

    print("3. Applying Gaussian distance decay...")
    df_acc["pop_dist_gaussian_with_r"] = (
        df_acc["Total Population"] * df_acc["r_ij"] * gaussian_decay_modified(df_acc["Distance"], D0, EPSILON)
    )

    print("4. Calculating V_j (weighted total visits per POI)...")
    Vj_df = (
        df_acc.groupby("PLACEKEY", as_index=False)["pop_dist_gaussian_with_r"]
              .sum()
              .rename(columns={"pop_dist_gaussian_with_r": "V_j"})
    )
    df_acc = pd.merge(df_acc, Vj_df, on="PLACEKEY", how="left")

    print("5. Calculating partial accessibility A_i^G...")
    df_acc["partialA"] = (
        (df_acc["RAW_VISIT_COUNTS"] / df_acc["V_j"]) *  # S_j / V_j
        df_acc["r_ij"] *                                # * r_ij
        gaussian_decay_modified(df_acc["Distance"], D0, EPSILON)  # * decay
    )

    print("6. Aggregating total accessibility per CBG...")
    AiG_df = (
        df_acc[~df_acc["partialA"].isin([np.inf, -np.inf])]
              .groupby("CBG_ID", as_index=False)["partialA"]
              .sum()
              .rename(columns={"partialA": "Accessibility"})
    )
    df_acc = pd.merge(df_acc, AiG_df, on="CBG_ID", how="left")

    print(f"7. Saving results → {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(df_acc, f)

    print(f"✅ Done. Saved {len(df_acc)} rows.")
