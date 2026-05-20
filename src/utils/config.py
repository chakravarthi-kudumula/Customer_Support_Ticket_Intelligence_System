from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def project_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path


def portable_project_path(path_value: str | Path | None) -> Path | None:
    """Resolve project artifact/data paths that may come from another machine."""
    if path_value is None:
        return None

    path = Path(path_value)
    if path.exists():
        return path

    path_text = str(path_value)
    for folder in ("artifacts", "data", "configs"):
        marker = f"/{folder}/"
        if marker in path_text:
            suffix = path_text.split(marker, 1)[1]
            return project_path(Path(folder) / suffix)

    return path

