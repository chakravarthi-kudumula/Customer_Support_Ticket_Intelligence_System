from pathlib import Path

import pandas as pd

from src.utils.config import load_config, project_path


def load_complaints(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the CFPB complaints CSV into a pandas DataFrame."""
    config = load_config()
    path = Path(csv_path) if csv_path else project_path(config["paths"]["raw_complaints_csv"])

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CFPB CSV in data/raw/ or pass csv_path."
        )

    return pd.read_csv(path)

