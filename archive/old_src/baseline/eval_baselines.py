from __future__ import annotations
from pathlib import Path
import pandas as pd

def append_metrics_csv(rows: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
