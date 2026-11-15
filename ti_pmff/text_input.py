"""BERT text input generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ELEMENT_COLS, PROCESS_COL
from .schema import prepare_base_dataframe


def build_text_inputs(df: pd.DataFrame) -> list[str]:
    prepared = prepare_base_dataframe(df)
    texts = []
    for _, row in prepared.iterrows():
        parts = []
        for col in ELEMENT_COLS:
            value = row.get(col, 0)
            if pd.notna(value) and float(value) > 0:
                parts.append(f"{col.replace(' (wt%)', '')} {float(value):.6g}%")
        composition = ", ".join(parts)
        texts.append(f"Titanium alloy composition: {composition}. Process description: {row[PROCESS_COL]}")
    return texts


def write_bert_input_txt(df: pd.DataFrame, output_txt: Path) -> Path:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as handle:
        for line in build_text_inputs(df):
            handle.write(line + "\n")
    return output_txt

