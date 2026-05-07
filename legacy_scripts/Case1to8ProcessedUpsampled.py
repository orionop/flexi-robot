import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

case_ids = range(1, 9)
desired_points = 1000
pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
delta_cols = ['Δx', 'Δy', 'Δz', 'Δqx', 'Δqy', 'Δqz', 'Δqw']
interp_axes = ['x', 'y', 'z']

for case_id in case_ids:
    input_file = f"case{case_id}_processed.csv"
    output_file = f"case{case_id}_upsampled.csv"

    try:
        df = pd.read_csv(input_file, encoding='utf-8')

        # Handle index for interpolation
        if 'Time' not in df.columns or df['Time'].nunique() <= 1:
            df["Time"] = df.index

        df = df.sort_values("Time").dropna(subset=pose_cols)

        if len(df) < 2:
            print(f"⚠️ Case {case_id}: Not enough valid data to interpolate.")
            continue

        t_old = df["Time"].values
        t_new = np.linspace(t_old.min(), t_old.max(), desired_points)

        up_df = pd.DataFrame({"Time": t_new})

        for col in pose_cols + delta_cols:
            if col in df.columns:
                try:
                    interp_fn = interp1d(t_old, df[col], kind='linear', bounds_error=False, fill_value="extrapolate")
                    up_df[col] = interp_fn(t_new)
                except Exception as e:
                    print(f"⚠️ Could not interpolate {col} in case {case_id}: {e}")

        # ST_A interpolated columns
        for axis in interp_axes:
            colname = f"ST_A_{axis}_interp"
            if colname in df.columns:
                try:
                    interp_fn = interp1d(t_old, df[colname], kind='linear', bounds_error=False, fill_value="extrapolate")
                    up_df[colname] = interp_fn(t_new)
                except Exception as e:
                    print(f"⚠️ Skipped ST_A interpolation for {axis}: {e}")

        # Save with correct UTF-8 encoding and header
        up_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ Case {case_id} upsampled and saved to {output_file}")

    except Exception as e:
        print(f"❌ Error processing case {case_id}: {e}")
