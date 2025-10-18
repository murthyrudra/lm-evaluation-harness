import os
import json
import pandas as pd

base_dir = "output"
data = []

# task prefixes you care about
valid_prefixes = ["bbk", "bba", "bbl", "bbf"]

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith("results") and file.endswith(".json"):
            path = os.path.join(root, file)
            try:
                with open(path, "r") as f:
                    j = json.load(f)

                # model name = first-level subfolder under "output"
                parts = os.path.relpath(path, base_dir).split(os.sep)
                model_name = parts[0] if len(parts) > 0 else "unknown"

                # iterate through tasks
                for task, metrics in j.get("results", {}).items():
                    if task in valid_prefixes:
                        acc = metrics.get("acc,none", None)
                        stderr = metrics.get("acc_stderr,none", None)
                        data.append({
                            "model": model_name,
                            "task": task,
                            "accuracy": acc,
                        })

            except Exception as e:
                print(f"Error reading {path}: {e}")

# Create dataframe
df = pd.DataFrame(data)

# Pivot so each task becomes a column
pivot_df = df.pivot_table(index="model", columns="task", values="accuracy", aggfunc="mean")

# Reset index for nice printing
pivot_df = pivot_df.reset_index()

# Reorder columns according to prefix order
cols = ["model"]
for prefix in valid_prefixes:
    # include all columns that start with this prefix (in sorted order)
    cols.extend(sorted([c for c in pivot_df.columns if c.startswith(prefix)]))
pivot_df = pivot_df[cols]

# Round to 4 decimal places
pivot_df = pivot_df.round(4)


print(pivot_df.to_string(index=False))

