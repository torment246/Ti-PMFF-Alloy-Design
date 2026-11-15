"""Project-relative input and output paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path, project_root=PROJECT_ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_input_path(path, project_root=PROJECT_ROOT):
    return resolve_path(path, project_root)


def resolve_output_path(path, project_root=PROJECT_ROOT):
    return resolve_path(path, project_root)
